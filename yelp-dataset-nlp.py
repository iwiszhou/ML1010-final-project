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
    filename = Path(__file__).parent / './yelp_dataset/review.json'
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

# Check number of rows
print(df.shape[0])

# List all columns
print(df.columns)

# Check data-set
print(df.head().values)

# Check na values
print(df.isnull().values.sum())


# There are not NA value in this data-set.


# Create a class(label) column
def get_class_label_value(row):
    if row["stars"] >= 3:
        return "Positive"
    return "Negative"


if not os.path.isfile("review.csv"):
    df["class"] = df.apply(get_class_label_value, axis=1)

    # Create new data frame
    filter_df = df[['text', 'class']]
    print(filter_df.head(1).values)
    print(filter_df.shape[0])
    print(filter_df.columns)

    # Export to csv
    filter_df.to_csv("review.csv", encoding='utf-8', index=False)
else:
    # Import csv
    filter_df = pd.read_csv("review.csv", encoding='utf-8')

'''
STEP 2 - Clean data / Text pre-processing
'''

# Text Normalization
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

filter_df["norm_text"] = normalize_corpus(filter_df["text"])

# Check the result
print(filter_df["norm_text"].describe())
print(filter_df.head(1))

#     # Export to csv
#     filterDf.to_csv("norm_review.csv", encoding='utf-8', index=False)
# else:
#     # Import from csv
#     filterDf = pd.read_csv("norm_review.csv", encoding='utf-8')


### As the result, the norm text is still have some words not fully converted to what we want.
### Such as, 'checked', 'costs', we expected those should stem correctly.
### Next, let's try library spacy.

# Text Normalization - part 2
nlp = spacy.load("en_core_web_sm")


def spacy_norm_doc(doc):
    doc = nlp(doc)
    # remove SYM, PUNCT, PRON and
    doc = filter(lambda d: d.pos_ != "SYM" and d.pos_ != "PUNCT" and d.pos_ != "PRON", doc)
    # extra lemma_ only
    lemma_text = [token.lemma_ for token in doc]
    # filter stopwords out of document
    filter_text = [token for token in lemma_text if token not in stop_words]
    # remove duplicate token
    unique_text = list(set(filter_text))
    return " ".join(unique_text)


filter_df["norm_text"] = spacy_norm_doc(filter_df["text"])

# Check the result
print(filter_df["norm_text"].describe())
print(filter_df.head(1))


# ## Tokenization
# tokenizer = nltk.tokenize.TreebankWordTokenizer()
# tokens = tokenizer.tokenize(text)
#
# ## Token Normalization

# ### Stemming (only apply for Verb)
# stemmer = nltk.stem.PorterStemmer()
# " ".join(stemmer.stem(token) for token in tokens)
#
# ### Lemmaztization (not able to hanld Verb)
# stemmer2 = nltk.stem.WordNetLemmatizer()
# " ".join(stemmer2.lemmatize(token) for token in tokens)

### Normazlizing capital letters ( lower case only first word in the sentence )

## Feature extraction from text

### TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# texts = ["",""]
# tfidf = TfidfVectorizer(min_df=2,max_df=0.5, ngram_range=(1,2))
# features = tfidf.fit_transform(texts)
# pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(filter_df["norm_text"])
tv_matrix = tv_matrix.toarray()
vocab = tv.get_feature_names()
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

## Model

### Logistic regression of 1-gram with TF-IDF

### Logistic regression of 2-gram with TF-IDF
