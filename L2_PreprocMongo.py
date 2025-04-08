from pymongo import ASCENDING, DESCENDING
from pymongo import MongoClient, errors
import csv
import time
import yaml
from pymongo import InsertOne
import numpy as np
import pandas as pd

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def get_database():
    """Establish a connection to the MongoDB database."""
    try:
        # Provide the MongoDB URI here (modify as per your setup)
        mongo_uri = config['mongodb_raw']['uri']
        client = MongoClient(mongo_uri)

        # Connect to a specific database
        db = client['ecomdb']
        print("Connected to MongoDB successfully.")
        return db
    except Exception as e:
        print("Error connecting to MongoDB:", e)
        raise

def get_collection(db, collection_name):
    """Fetch a collection from the database."""
    return db[collection_name]

def delete_collection(db, collection_name):
    try:
        print("Deleting contents before inserting.")
        collection = db[collection_name]
        result = collection.delete_many({})  # Empty filter deletes everything
        if result.deleted_count==0:
            print("No data to delete.")
        else:
            print(f"Deleted {result.deleted_count} documents.")
        print("Deletion completed.\n")
    except Exception as e:
        print("Error deleting from collection:", e)

def backup_before_insert(db, collection_name):
    try:
        print("\nStarting Backup.")
        collection = db[collection_name]
        # Retrieve all existing data from the original collection
        existing_data = list(collection.find())

        backup_date = time.strftime("%Y%m%d")  # Format: YYYYMMDD
        backup_collection_name = "backup_" + backup_date
        backup_collection = get_collection(db, backup_collection_name)
        # Insert the existing data into the backup collection
        if existing_data:
            backup_collection.insert_many(existing_data)
            print(f"Moved {len(existing_data)} documents to the backup collection.")
            print("Backup Completed.\n")
    except Exception as e:
        print(f"Error occurred: {e}")

def insert_file_into_collection_raw(collection_name, file_path):
    """Insert the contents of a file into a specified collection."""
    try:
        # Get the database connection
        db = get_database()
        collection = get_collection(db, collection_name)
        backup_before_insert(db, collection_name)
        delete_collection(db, collection_name)
        # create_indexes(db, collection_name)
        print("Inserting data to the collection.")
        with open(file_path,  'r', encoding='utf-8', errors='replace') as file:  # Explicit encoding
            reader = csv.DictReader(file)  # Use DictReader to read rows as dictionaries
            rows = list(reader)  # Convert the reader object to a list of rows
        if rows:
            # Insert the data in bulk, using smaller batches to avoid timeouts
            batch_size = 1000  # Insert 1000 documents at a time
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                bulk_operations = [InsertOne(doc) for doc in batch]
                collection.bulk_write(bulk_operations)
            # collection.insert_many(rows)
            print(f"CSV file '{file_path}' inserted successfully into collection '{collection_name}'.")
        else:
            print(f"CSV file '{file_path}' is empty. No data inserted.")
        print("Data Insertion completed.\n")
        try:
            df = extract_data(collection)
        except Exception as e:
            print("Error Preprocessing and Inserion of data:", e)
    except Exception as e:
        print("Error inserting CSV file into collection:", e)


# Function to preprocess the data
def preprocess_data(df):
    """Preprocess the data."""
    # print("Missing values before cleaning:\n", df.isna().sum())

    print("\nPreprocessing Raw Data.")
    # Function to clean numeric columns
    def clean_numeric(column):
        return pd.to_numeric(df[column].astype(str).str.replace("₹", "").str.replace(",", ""), errors='coerce')

    # Clean numeric columns
    df["ratings"] = clean_numeric("ratings")
    df["no_of_ratings"] = clean_numeric("no_of_ratings")
    df["discount_price"] = clean_numeric("discount_price")
    df["actual_price"] = clean_numeric("actual_price")

    # Fill missing values with mean or corresponding values
    df["ratings"].fillna(df["ratings"].mean(), inplace=True)
    df["no_of_ratings"].fillna(df["no_of_ratings"].mean(), inplace=True)
    df["discount_price"].fillna(df["actual_price"], inplace=True)
    df["actual_price"].fillna(df["discount_price"], inplace=True)

    # Drop unnecessary columns
    df.drop(columns=["Unnamed: 0", "image", "link"], inplace=True)

    # Discount Percentage
    df["discount_percentage"] = ((df["actual_price"] - df["discount_price"]) / df["actual_price"]) * 100

    # Price Ratio
    df["price_ratio"] = df["discount_price"] / df["actual_price"]

    # Popularity Score
    df["popularity_score"] = df["ratings"] * np.log1p(df["no_of_ratings"])

    # Price Difference
    df["price_difference"] = df["actual_price"] - df["discount_price"]

    # Log transformation of ratings and number of ratings
    df["log_no_of_ratings"] = np.log1p(df["no_of_ratings"])

    # Category Encoding
    df["main_category_encoded"] = df["main_category"].astype('category').cat.codes
    df["sub_category_encoded"] = df["sub_category"].astype('category').cat.codes
    print("Preprocessing Raw Data Completed.")
    return df

def extract_data(collection):
    # Extract data from the source collection
        try:
            print("\nExtracting data from source.")
            data = list(collection.find())
            # Convert the data into a DataFrame
            df = pd.DataFrame(data)
            # Preprocess the data
            print("Extraction Completed.")
            df = preprocess_data(df)
            status= insert_to_tgt_db(df)
            if status:
                print("\nExecution Completed.")
        except Exception as e:
            print(f"Error extracting data from the source: {e}")
            exit(1)

# Function to insert data in smaller batches
def insert_in_batches(collection, data, batch_size=1000):
    """Insert data into MongoDB in smaller batches."""
    try:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            bulk_operations = [InsertOne(doc) for doc in batch]
            collection.bulk_write(bulk_operations)
        print(f"Inserted {len(data)} documents successfully.")
    except Exception as e:
        print(f"Error inserting data in batches: {e}")

# Create indexes in Mongodb database 
def create_indexes(db, collection_name):
    try:
        print("Creating Indexes.")
        collection = get_collection(db, collection_name)
        # 1. Compound Indexes : queried t0gether
        collection.create_index([('main_category', 1), ('sub_category', 1)], name="main_category_sub_category_index") 
        collection.create_index([('ratings', -1), ('no_of_ratings', 1)], name="ratings_no_of_ratings_index")  

        # 2. Unique Index : Ensures product names are unique
        collection.create_index([('link', 1)], unique=True, name="unique_product_name_index")  

        # 3. Text Index : support text search functionality
        collection.create_index([('name', 'text')], name="product_name_text_search_index")

        # collection.drop_index('ratings_-1_no_of_ratings_1') 
        print("Indexes created.\n")
    except Exception as e:
        print("Error creating indexes", e)


def insert_to_tgt_db(df):
    print("\nInserting data into Target database.")
    try:
        mongo_uri = config['mongodb']['uri']
        client = MongoClient(mongo_uri)
        # Connect to a specific database
        tgt_db = client['ecomdb']
        col = config['mongodb']['collection_test']
        tgt_collection = tgt_db[col]
        # print("Connected to Tgt MongoDB successfully.")
    except Exception as e:
        print("Error connecting to Tgt MongoDB:", e)
        raise
    try:
        preprocessed_data = df.to_dict(orient='records')
        # Insert the data into the destination collection
        tgt_collection.insert_many(preprocessed_data)
        insert_in_batches(tgt_collection, preprocessed_data)
        print(f"Data successfully inserted into collection '{tgt_collection}'.")
        return True
    except Exception as e:
        print("Error inserting to Target MongoDB:", e)
        return False
