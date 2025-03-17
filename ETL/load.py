import os
import pandas as pd
import logging

# Configure logging

def load_data(df, data_dir, file_name):
    """
    Load a DataFrame to a CSV file in the specified data directory.
    Appends new data and prevents duplicate entries.

    Args:
    df (pandas.DataFrame): The DataFrame to be saved.
    data_dir (str): The directory where the CSV file will be saved.
    file_name (str): The name of the file to be saved (without extension).
    """

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Create the full file path
    file_path = os.path.join(data_dir, f"{file_name}.csv")

    try:
        # Check if file exists
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)

            # Merge old and new data, dropping duplicates
            combined_df = pd.concat([existing_df, df]).drop_duplicates()

            # Save the updated DataFrame
            combined_df.to_csv(file_path, index=False)
            logging.info(f"✅ Updated {file_name}.csv with new data.")
        else:
            # Save new data as a fresh CSV
            df.to_csv(file_path, index=False)
            logging.info(f"✅ Created new {file_name}.csv file.")
    
    except Exception as e:
        logging.error(f"❌ Failed to load {file_name}.csv: {e}")

