
import time
import csv
import logging
from datetime import datetime

import streamlit as st
from openai import OpenAI

# Configuration imports
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_config as cfg
# OpenAI API Key Handling
deploy = cfg.deploy
if deploy:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
else:
    from dotenv import load_dotenv
    load_dotenv('.env')

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# OpenAI Embeddings for VectorDB (used internally, not user-provided)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=EMBEDDING_API_KEY)

# Define ChromaDB location
persist_directory = "./chroma_langchain_db"

# Lazy load ChromaDB collections
def get_vectordb(selected_collection):
    """Load the appropriate ChromaDB collection based on user selection."""
    collection_name = "alcon_collection_financial_statements_annually" if selected_collection == "Annually" else "alcon_collection_financial_statements_quarterly"
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)

# Retrieve relevant entries from ChromaDB
def find_relevant_entries_from_chroma_db(query, selected_collection):
    """Search ChromaDB for the most relevant financial documents."""
    vectordb = get_vectordb(selected_collection)
    results = vectordb.similarity_search_with_score(query, k=3)

    filtered_results = [doc.page_content for doc, score in results if score > 0.5]  # Skip low-relevance results
    return "\n".join(filtered_results) if filtered_results else "No relevant financial data found."

# Generate GPT-based response
def generate_gpt_response(user_query, chroma_result, client):
    """Generate response using GPT model with retrieved financial data as context."""
    current_year = datetime.now().year
    last_quarter = (datetime.now().month - 1) // 3  # Auto-detect last quarter

    combined_prompt = f"""User query: {user_query}

    You are a financial analyst at ALCON Inc. Provide an answer based on:
    {chroma_result}

    The current year is {current_year}, and the last available quarter is Q{last_quarter}.
    Format your response clearly and concisely.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial assistant providing data-driven insights."},
            {"role": "user", "content": combined_prompt}
        ]
    )
    return response.choices[0].message.content

# Log response times for monitoring
def log_response_time(query, response_time, is_first_prompt):
    """Log chatbot performance metrics for analysis."""
    csv_file = 'responses.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Query', 'Response Time (seconds)', 'Is First Prompt'])
        writer.writerow([datetime.now(), query, f"{response_time:.2f}", "Yes" if is_first_prompt else "No"])

# Query processing pipeline
def query_interface(user_query, is_first_prompt, selected_collection, client):
    '''
    Process user query and generate a response using GPT model and relevant information from the database.
    '''
    start_time = time.time()  # ⏱️ Start measuring response time

    #  Handle competitor-related queries
    if 'competitors' in user_query.lower():
        competitor_list = [ticker for ticker in cfg.tickers if ticker != "ALC"]
        if competitor_list:
            user_query = user_query.replace("competitors", f"competitors including {', '.join(competitor_list)}")

    #  Step 1: Find relevant financial data from ChromaDB
    chroma_result = find_relevant_entries_from_chroma_db(user_query, selected_collection)

    #  If no relevant data is found, return early (prevent unnecessary GPT call)
    if not chroma_result or "No relevant financial data found" in chroma_result:
        return " No relevant financial data available for your query."

    try:
        # 🧠 Step 2: Generate GPT-augmented response
        gpt_response = generate_gpt_response(user_query, chroma_result, client)
    except Exception as e:
        logging.error(f" Error while generating GPT response: {e}")
        return " There was an error processing your request. Please try again later."

    # 📊 Step 3: Log response time for analytics
    logging_response_time = True
    if logging_response_time:
        response_time = time.time() - start_time
        log_response_time(user_query, response_time, is_first_prompt)

    # 🔄 Step 4: Return the generated response
    return gpt_response


# Streamlit Chat Interface
def display_chatbot():
    st.title(" Alcon Financial Chatbot")

    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not api_key:
        st.info("🔑 Please enter your OpenAI API key to continue.", icon="🔒")
        return

    client = OpenAI(api_key=api_key)
    selected_collection = st.radio("Select Data Period:", ("Annually", "Quarterly"))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a financial question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing data..."):
            response = query_interface(prompt, len(st.session_state.messages) == 1, selected_collection, client)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    display_chatbot()

if __name__ == "__main__":
    main()
