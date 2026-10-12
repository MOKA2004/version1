import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.hubble")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGO_DATABASE", "hubble_employee_db")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[DATABASE_NAME]
employees_collection = db["employees"]
employee_directory_collection = db["employee_directory"]
