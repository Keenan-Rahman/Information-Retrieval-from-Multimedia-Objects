from pymongo import MongoClient
from sklearn.datasets import fetch_olivetti_faces
from bson.binary import Binary

import numpy as np
import pickle

print("\nFetching Olivetti Faces Dataset from sklearn\n")

# Fetch dataset form sklearn library
olivetti = fetch_olivetti_faces()

# create database connection on mongodb server with port=27017, appname = MongoDB Compass, and no username and password
client = MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
db = client.admin

# create a database named "mwdb_database"
mwdb_database = client["mwdb_database"]

# create a table named "olivetti_faces"
olivetti_faces_col = mwdb_database["olivetti_faces"]
count = 0
print("\nData Fetch Successful\n")
print("\nSaving Dataset to MongoDB Server\n")

# loop on all the images in the dataset and save these images in the database. All images are converted using
# pickle.dump function and then converted to Binary values.
for j in np.array(olivetti.images):
    olivetti_faces_col.insert_one({"_id": count, "image_matrix": Binary(pickle.dumps(j, protocol=2))})
    count += 1
print("\nData Saved to Server\n")
