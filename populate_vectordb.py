import os
from dotenv import load_dotenv
import pandas as pd
import logging
import shutil
import streamlit as st
import app_config as cfg
# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
deploy = cfg.deploy
if deploy:
    OPENAI_API_KEY = st. secrets["OPENAI_API_KEY"]
else:
    load_dotenv('.env')  # looks for .env in Python script directory unless path is provided
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

# Document locations (relative to this py file)
folder_paths = ['data']
DB_PATH = "./chroma_langchain_db"  # Centralized path for the database

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import json
import hashlib
import pandas as pd
import logging
from langchain_core.documents import Document

# Define paths
HASH_FILE = "csv_hashes.json"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_text_hash(text):
    """Compute SHA-256 hash of a text chunk to track changes."""
    return hashlib.sha256(text.encode()).hexdigest()


def load_existing_hashes():
    """Load previously stored CSV hashes to detect changes."""
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, "r") as f:
                data = f.read().strip()
                return json.loads(data) if data else {}  # Handle empty file
        except json.JSONDecodeError:
            logging.warning(" Hash file is corrupted, resetting hash data.")
            return {}
    return {}


def save_hashes(updated_hashes):
    """Save updated hashes after processing new CSVs."""
    with open(HASH_FILE, "w") as f:
        json.dump(updated_hashes, f, indent=4)


def process_csv_files(folders):
    """
    Process CSV files, extract financial data, and store embeddings only for new/modified data.
    Uses hashing to track changes.
    """
    all_docs_annually = []
    all_docs_quarterly = []
    existing_hashes = load_existing_hashes()
    new_hashes = {}

    for folder in folders:
        if not os.path.exists(folder):
            logging.warning(f" Folder '{folder}' does not exist.")
            continue

        logging.info(f" Processing folder: {folder}")
        csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

        for file in csv_files:
            file_path = os.path.join(folder, file)
            logging.info(f" Processing CSV file: {file_path}")

            df = pd.read_csv(file_path)

            for col in df.columns:
                if col not in ["Date", "Symbol", "Quarter", "Year"]:
                    content = []
                    for _, row in df.iterrows():
                        content.append(
                            f"Date: {row['Date']}, Symbol: {row['Symbol']}, Quarter: {row['Quarter']}, Year: {row['Year']}, {col}: {row[col]}"
                        )

                    full_content = "\n".join(content)
                    chunk_hash = compute_text_hash(full_content)

                    # Check if this metric has changed
                    if existing_hashes.get(file, {}).get(col) == chunk_hash:
                        logging.info(f" No changes detected for {file} → {col}. Skipping...")
                        continue  # Skip embedding if content is unchanged

                    # Store updated hash
                    if file not in new_hashes:
                        new_hashes[file] = {}
                    new_hashes[file][col] = chunk_hash

                    metadata = {"source": file, "column": col}
                    doc = Document(page_content=full_content, metadata=metadata)

                    if "annual" in file.lower():
                        all_docs_annually.append(doc)
                    elif "quarterly" in file.lower():
                        all_docs_quarterly.append(doc)

                    logging.info(f"🔄 Processed column '{col}' from {file}.")

    # Save updated hashes
    if new_hashes:
        existing_hashes.update(new_hashes)
        save_hashes(existing_hashes)

    logging.info(f" Total annual documents created: {len(all_docs_annually)}")
    logging.info(f" Total quarterly documents created: {len(all_docs_quarterly)}")

    return all_docs_annually, all_docs_quarterly


import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DB_PATH = "./chroma_langchain_db"

def update_vector_db(docs_annually, docs_quarterly):
    """
    Updates ChromaDB only for new/modified documents instead of deleting the entire DB.
    """
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")

        client = chromadb.PersistentClient(path=DB_PATH)

        # Load existing vector collections
        alcon_vectorstore_annually = Chroma(
            client=client,
            collection_name="alcon_collection_financial_statements_annually",
            embedding_function=embeddings,
        )
        alcon_vectorstore_quarterly = Chroma(
            client=client,
            collection_name="alcon_collection_financial_statements_quarterly",
            embedding_function=embeddings,
        )

        # If database is empty, fully populate it
        if alcon_vectorstore_annually._collection.count() == 0 and alcon_vectorstore_quarterly._collection.count() == 0:
            logging.info(" ChromaDB is empty. Performing full embedding...")
            alcon_vectorstore_annually.add_documents(documents=docs_annually)
            alcon_vectorstore_quarterly.add_documents(documents=docs_quarterly)
            logging.info(" Full vector database update completed.")
        else:
            # Only update new/modified records
            if docs_annually:
                logging.info(f" Adding {len(docs_annually)} annual documents to ChromaDB...")
                alcon_vectorstore_annually.add_documents(documents=docs_annually)

            if docs_quarterly:
                logging.info(f" Adding {len(docs_quarterly)} quarterly documents to ChromaDB...")
                alcon_vectorstore_quarterly.add_documents(documents=docs_quarterly)

            logging.info(" Incremental vector database update completed.")
    
    except Exception as e:
        logging.error(f" Error updating ChromaDB: {str(e)}")
        raise

# Handle API Key Securely
if deploy:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        raise ValueError(" ERROR: OpenAI API Key is missing in Streamlit secrets.toml!")
else:
    load_dotenv('.env')  
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError(" ERROR: OpenAI API Key is missing in .env file!")

# Define Paths
DB_PATH = "./chroma_langchain_db"
folder_paths = ['data']

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """
    Main function to process CSV files and update the vector database efficiently.
    Only updates embeddings if new data is detected.
    """
    try:
        logging.info(" Starting vector database update process...")

        # Step 1: Process CSV Files (Extract & Transform)
        all_docs_annually, all_docs_quarterly = process_csv_files(folder_paths)

        # Step 2: Check if there are new/modified documents
        if not all_docs_annually and not all_docs_quarterly:
            logging.info(" No new updates detected in financial data. Skipping database update.")
            return

        # Step 3: Update Vector Database with new data
        update_vector_db(all_docs_annually, all_docs_quarterly)

        logging.info("🎉 Vector database update completed successfully!")

    except ValueError as e:
        logging.error(f" CRITICAL ERROR: {str(e)}")
    except Exception as e:
        logging.error(f" An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()


