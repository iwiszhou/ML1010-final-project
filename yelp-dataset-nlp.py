"""
Source code - https://github.com/iwiszhou/ML1010-final-project

Dataset - https://www.yelp.com/dataset/download

This is a Yelp data-set. I would use this data-set to do a sentiment analysis.
I would build a model to predict the review either positive or negative.
This is a big data-set. Firstly, I would try to extra the review data and create a simple data-set,
which only contain Review & Rating. After that, I would create a new column which is Class.
Class column is either Positive or Negative. If Rating is grater than 3, I would mark Class to Positive. Otherwise,
Negative. If I have more time at the end, I would introduce one more value to Class column which is Neutral ( when
Rating is equal to 3 )
"""

# Import libraries
import pandas as pd
import json
import numpy as np
import re
import nltk
import sqlite3
import matplotlib.pyplot as plt

nltk.download('stopwords')

# Read data from file
# NOTE - the data-set is too big. I have already to several time, my computer crash. So that, I would start with first
# 10000 rows. I would increase the data-set size when training the model.
filename = "/Users/iwiszhou/Documents/Machine Learning/ML1010/ML1010-final-project/yelp_dataset/review.json"
rowCount = 0
rowLimit = 10000
df = []
with open(filename, 'r') as f:
    for line in f:
        df.append(json.loads(line))
        rowCount = rowCount + 1
        if rowCount > rowLimit:
            break
df = pd.DataFrame(df)

# Check data-set
print(df.head())

# Check number of rows
count_row = df.shape[0]
print(count_row)

# List all columns
print(df.columns)


# Create a class column
def get_class_label_value(row):
    if row["stars"] >= 3:
        return "Positive"
    return "Negative"


df["class"] = df.apply(lambda row: get_class_label_value(row), axis=1)

# Create new data frame
filterDf = df[['business_id', 'text', 'class']]
print(filterDf.head().to_string())
print(filterDf.shape[0])
print(filterDf.columns)

# Text pre-processing
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


normalize_corpus = np.vectorize(normalize_document)


# Create a clean text column

def get_clean_text_value(row):
    if row["text"]:
        return normalize_corpus(row["text"])
    return ""


filterDf["clean_text"] = df.apply(lambda row: get_clean_text_value(row), axis=1)

# Exploring Column Summaries
print(filterDf.sample(2, random_state=42).to_string())
print(filterDf.dtypes)
print(len(filterDf))
print(filterDf.describe(include=np.object).transpose().to_string())



# Save data to database
#con = sqlite3.connect('yelp.db')
#filterDf.to_sql("filter_reviews", con)
