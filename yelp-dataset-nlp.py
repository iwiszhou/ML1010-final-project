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
from pathlib import Path
import os
import spacy

# Download stopwords if not existing
nltk.download('stopwords')

# Set col to max width
pd.set_option('display.max_colwidth', -1)

# Database name & tables' name
db_name = "yelp.db"
table_names = {
    "reviews": "reviews",
    "clean_reviews": "clean_reviews"
}

# Helper functions
def get_absolute_path(file_name):
    return Path(__file__).parent / file_name


# Save data to database
def save_to_db(dataFrame, tableName):
    con = sqlite3.connect(db_name)
    dataFrame.to_sql(tableName, con)
    con.close()


# Get data (dataframe format) from database by table name
def get_table_by_name(tableName):
    con = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM " + tableName + ";", con)
    con.close()
    return df


# Read data from file
# NOTE - the data-set is too big. I have already to several time, my computer crash. So that, I would start with first
# 10000 rows. I would increase the data-set size when training the model.
def load_json():
    filename = get_absolute_path('./yelp_dataset/review.json')
    row_count = 0
    row_limit = 10000
    df = []
    with open(filename, encoding="utf8") as f:
        for line in f:
            df.append(json.loads(line))
            row_count = row_count + 1
            if row_count > row_limit:
                break
    df = pd.DataFrame(df)
    return df


'''
STEP 1 - Gather data
'''

# Get data from database or json file
if os.path.isfile(db_name) and os.path.getsize(db_name) > 100:
    df = get_table_by_name(table_names["reviews"])
else:
    df = load_json()
    save_to_db(df, table_names["reviews"])

# Top 5 records
print(df.head().values)

# Shape of dataframe
print(df.shape)

# View data information
print(df.info())

# Check na values
print(df.isnull().values.sum())

"""
There are not NA value in this data-set. Next, let's create a new column to store our Class/Label value,
which depennds on our 'stars' column, if 'stars' is great than 3, Class/Label is 1 - 'Positive'. Otherwise,
it is 0 - 'Negative'
"""


# Create a class(label) column
def get_class_label_value(row):
    if row["stars"] >= 3:
        return 1
    return 0


review_file_path = get_absolute_path("review.csv")

if not os.path.isfile(review_file_path):
    df["class"] = df.apply(get_class_label_value, axis=1)

    # Create new data frame
    filter_df = df[['class', 'text']]
    print(filter_df.head(1).values)
    print(filter_df.shape[0])
    print(filter_df.columns)

    # Export to csv
    filter_df.to_csv(review_file_path, encoding='utf-8', index=False)
else:
    # Import csv
    filter_df = pd.read_csv(review_file_path, encoding='utf-8')

'''
STEP 2 - Clean data / Text pre-processing
'''

"""
First of all, let's balance the data
"""
balance_review_file_path = get_absolute_path("balance_review.csv")

if not os.path.isfile(balance_review_file_path):
    # num of Positive record
    print(filter_df.loc[filter_df["class"] == 1].count())

    # num of Negative record
    print(filter_df.loc[filter_df["class"] == 0].count())

    # balance the data
    balance_data_count = 10
    n_df = filter_df.loc[filter_df["class"] == 0][:balance_data_count]
    # number of negative rows
    print("Number of negative should be 100. Actual is ", len(n_df.loc[n_df["class"] == 0]))
    print("Number of positive should be 0. Actual is ", len(n_df.loc[n_df["class"] == 1]))

    p_df = filter_df.loc[filter_df["class"] == 1][:balance_data_count]
    # number of positive rows
    print("Number of positive should be 100. Actual is ", len(p_df.loc[p_df["class"] == 1]))
    print("Number of negative should be 0. Actual is ", len(p_df.loc[p_df["class"] == 0]))

    # merge positive and negative together to become a balance data
    filter_df = n_df.append(p_df)

    filter_df.to_csv(balance_review_file_path, encoding='utf-8', index=False)
else:
    # Import csv
    filter_df = pd.read_csv(balance_review_file_path, encoding='utf-8')

"""
Secondly, we would use NLTK method to normalize our corpus.
"""

# Text Normalization - using NLTK
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

# filter_df["norm_text"] = normalize_corpus(filter_df["text"])
#
# # Check the result
# print(filter_df["norm_text"].describe())
# print(filter_df.head(1))

"""
As the result, the norm text is still have some words not fully converted to what we want.
Such as, 'checked', 'costs', we expected those should stem correctly.
Next, let's try library Spacy, which provide all lots of helper method for us to normalize our corpus.
"""

# Text Normalization - using Spacy
nlp = spacy.load("en_core_web_sm")
white_list_pos = ["VERB", "PART", "NOUN", "ADJ", "ADV"]


def spacy_norm_text(text):
    # tokenizing
    doc = nlp(str(text))

    ret_set = set()

    # handle stop words, VERB, PART, ADJ, ADV and NOUN
    for token in doc:
        if not token.is_stop and token.text:  # remove stop words & empty string
            if token.pos_ in white_list_pos:  # if token is in white list, taking lemma_ instead
                ret_set.add(token.lemma_.lower().strip())

    # handle PROPN
    for token in doc.ents:
        ret_set.add(token.text)

    # convert to list
    unique_list = list(ret_set)

    return " ".join(unique_list)


norm_review_file_path = get_absolute_path("norm_review.csv")

if not os.path.isfile(norm_review_file_path):
    filter_df["norm_text"] = filter_df.apply(lambda row: spacy_norm_text(row["text"]), 1);

    # Export norm text to file
    filter_df.to_csv(norm_review_file_path, encoding='utf-8', index=False)
else:
    # Import norm text data frame
    filter_df = pd.read_csv(norm_review_file_path, encoding='utf-8')

# Check the result
print(filter_df["norm_text"].describe())
print(filter_df.head(1))

# Put all word in one set
words_set = set()
for doc in filter_df["norm_text"]:
    for token in doc.split():
        words_set.add(token)

'''
STEP 3 - Feature extraction from text
'''

"""
Using TF-IDF to convert text to vector
"""

### TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2)
tfidf = vectorizer.fit_transform(filter_df["norm_text"].values)

# convert to array
tfidf = tfidf.toarray()
print(tfidf.shape)  # 200 is our rows, 1186 is how many words

words = vectorizer.get_feature_names()

# plt.figure(figsize=[20,4])
# _ = plt.show(tfidf)

pd.DataFrame(tfidf, columns=words)

'''
STEP 4 - Build Models
'''

# Prepare the train and test dataset
from sklearn.model_selection import train_test_split

X = tfidf  # the features we want to analyze
y = filter_df['class'].values  # the labels, or answers, we want to test against

# split into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)

predict_ret = model.predict_proba(X_test)

# convert to Positive and Negative
y_predict = np.array([int(p[1] > 0.5) for p in predict_ret])

# accuracy
print(y_predict)
print(y_test)
print(np.sum(y_test == y_predict) / len(y_test))
