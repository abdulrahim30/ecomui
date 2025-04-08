# import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from elasticsearch import Elasticsearch
from pymongo import MongoClient, errors
import csv
import time
import yaml

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Build connection to Mongodb
def get_database():
    """Establish a connection to the MongoDB database."""
    try:
        # Access MongoDB credentials from the YAML file
        mongo_uri = config['mongodb']['uri']
        client = MongoClient(mongo_uri)

        # Connect to a specific database
        db = client['ecom']
        print("Connected to MongoDB successfully.")
        return db
    except Exception as e:
        print("Error connecting to MongoDB:", e)
        raise

# Retrieve collection name from Mongodb
def get_collection(db, collection_name):
    """Fetch a collection from the database."""
    return db[collection_name]

# Deleting a Mongodb collection
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

# Build connection to Elastic search
def es_conn():
    # Access ECS credentials from the YAML file
    cloud_id = config['ecs']['cloud_id']
    api_key = config['ecs']['api_key']
    # Connect to Elastic Cloud
    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key
    )
    index_name="data_ecom"
    return es, index_name

# Create a new index in Elastic search
def index_mongodb_data(collection_name):
    # index_name="data_ecom"
    es, index_name = es_conn()
    collection_mapping = {
    "settings": {
        "analysis": {
            "tokenizer": {
                "edge_ngram_tokenizer": {
                    "type": "edge_ngram",
                    "min_gram": 1,
                    "max_gram": 25,
                    "token_chars": ["letter", "digit"]
                }
            },
            "analyzer": {
                "edge_ngram_analyzer": {
                    "type": "custom",
                    "tokenizer": "edge_ngram_tokenizer"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "name": {
                "type": "text",
                "analyzer": "edge_ngram_analyzer"
            },
            "main_category": {
                "type": "keyword"
            },
            "sub_category": {
                "type": "keyword"
            },
            "image": {
                "type": "text"
            },
            "link": {
                "type": "keyword"
            },
            "ratings": {
                "type": "float"
            },
            "no_of_ratings": {
                "type": "float"
            },
            "discount_price": {
                "type": "float"
            },
            "actual_price": {
                "type": "float"
            },
    "discounted_price_1": {
      "type": "float"
    },
    "discounted_price_2": {
      "type": "float"
    },
    "discounted_price_3": {
      "type": "float"
    },
    "discounted_price_4": {
      "type": "float"
    },
    "discounted_price_5": {
      "type": "float"
    },
    "discounted_price_6": {
      "type": "float"
    },
    "discounted_price_7": {
      "type": "float"
    },
    "discount_percentage": {
      "type": "float"
    },
    "price_ratio": {
      "type": "float"
    },
    "popularity_score": {
      "type": "float"
    },
    "price_difference": {
      "type": "float"
    },
    "log_no_of_ratings": {
      "type": "float"
    },
    "main_category_encoded": {
      "type": "integer"
    },
    "sub_category_encoded": {
      "type": "integer"
    }
        }
    }}
    # Create the index with this mapping
    # response = es.indices.create(index=index_name, body=collection_mapping)
    # print(response)

    print("Fetching Data from Mongodb.")
    # Get the database connection
    db = get_database()

    # Get the collection
    collection = get_collection(db, collection_name)
    for doc in collection.find().skip(96001).batch_size(2000):
        # Prepare the document for indexing (without the '_id' field inside the document)
        document = {
    "name": doc["name"],
    "main_category": doc["main_category"],
    "sub_category": doc["sub_category"],
    "image": doc.get("image", ""),
    "link": doc.get("link", ""),
    "ratings": doc["ratings"],
    "no_of_ratings": doc["no_of_ratings"],
    "discount_price": doc["discount_price"],
    "actual_price": doc["actual_price"],
    "discounted_price_1": doc["discounted_price_1"],
    "discounted_price_2": doc["discounted_price_2"],
    "discounted_price_3": doc["discounted_price_3"],
    "discounted_price_4": doc["discounted_price_4"],
    "discounted_price_5": doc["discounted_price_5"],
    "discounted_price_6": doc["discounted_price_6"],
    "discounted_price_7": doc["discounted_price_7"],
    "discount_percentage": doc["discount_percentage"],
    "price_ratio": doc["price_ratio"],
    "popularity_score": doc["popularity_score"],
    "price_difference": doc["price_difference"],
    "log_no_of_ratings": doc["log_no_of_ratings"],
    "main_category_encoded": doc["main_category_encoded"],
    "sub_category_encoded": doc["sub_category_encoded"]
}
        
        # Use MongoDB _id as the document ID in Elastic, pass it as a parameter, not in the document
        response = es.index(index=index_name, id=str(doc["_id"]), document=document)
        print(f"Indexed document: {response}")
    print("Insert Successful!")    