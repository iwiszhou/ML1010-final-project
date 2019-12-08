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
from collections import Counter

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


'''
Read data from file
NOTE - the data-set is too big. I have already to several time, my computer crash. So that, I would start with first
10000 rows. I would increase the data-set size when training the model.
'''


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

# Get first n records data from csv or json file
first_n_records_file_path = get_absolute_path("first_n_records.csv")

if not os.path.isfile(first_n_records_file_path):
    df = load_json()
    # Export to csv
    df.to_csv(first_n_records_file_path, encoding='utf-8', index=False)
else:
    # Import csv
    df = pd.read_csv(first_n_records_file_path, encoding='utf-8')

'''
STEP 2 - Explore data
'''

'''
First fo all, let's take a look the dataset structure.
'''

# Let's take a look the size of the dataset & size for each columns
len(df)
df.count()

# Next, let's take a look list column names and datatypes
df.dtypes

'''
In this dataset, there is 9 columns and 10001 rows in total.
There are 5 columns contain String ( review_id, user_id, business_id, text, date )
The rest is either integer or number ( stars, useful, funny, cool )
'''

'''
Next, exploring the String/Category columns
'''
# Let's have a look for the text columns by selecting any 2 rows
df[["review_id", "user_id", "business_id", "text", "date"]].sample(2, random_state=40)

'''
It is very obviously review_id, user_id and business_id are unique String, 
which usually won't have any correlation with other columns. However, based on user_id and business_id,
we might also able to do some analysis, such as distribution of user groups in each store, or recommend
restaurants to user based on their reviews(However, for tis project, we only focus on sentiment analysis)
For text column, which is the actually reviews provide by user, which is what we need to analyze.
For date column, it is the date when user posted their review.

Let's continuous to look at the summary for the String/category features 
'''

# Describe categorical columns of type np.object
df[["review_id", "user_id", "business_id", "text", "date"]].describe(include=np.object).transpose()

'''
According to the result, we can find all review is identical. There is no duplicated review_id found.
For user_id column, there is only 9367 unique values, which means there is total 9367 customers in this dataset.
For business_id column, there is 4619 unique values, which means there is total 4619 business entities in this dataset.
And there is one business appears 79 times which business_id is Wxxvi3LZbHNIDwJ-ZimtnA
For text column, all records is unique, which is no duplicate records. This is also match what we found in review_id 
column, which is no duplicated review. So that, text column and review_id should have a 1:1 relationship.
For date column, there is 9988 unique values, which mean some customers post their review on the same date.

Next, let's take a look the numerical columns
'''

'''
Exploring the numerical columns
'''

# Describe numerical columns
df[["stars", "useful", "funny", "cool"]].describe().transpose()
'''
Each columns contains the same number count.
For stars column, the average value is 3.7 and range is from 1-5.
For useful column, the average value is 1.29 and range is from 0-91.
For funny column, the average value is 0.45 and rage is 0-42. In this column, range is up to 42, however, 
the average is only 0.45. It is possible most of value is 0.
For cool column, the average value is 0.55 and rage is 0-86. Same as funny column, we guess most value is 0.
'''

# Now, let's take a look how many 0 value in funny and cool columns
len(df.loc[df["funny"] == 0])
len(df.loc[df["funny"] == 0])
len(df[(df["cool"] == 0) & (df["funny"] == 0)])
'''
There is 7969 rows contains 0 value in funny column. And there is 7588 rows contains 0 value in cool column.
There is 6874 rows both funny and cool are 0, which is more than 68% of the whole dataset.
In this case, we won't consider those two columns in our analysis.
'''

'''
Visualizing
'''

# Plot the stars columns
df[["stars"]].plot(kind="box", vert=False, figsize=(6, 2))
# According to the graph, it tells us the mean is 4. And most of values are between 3-5.

# Let's group by business id and see what is the average rating for each business
df_gby_business = df.groupby("business_id", as_index=False)['stars'].mean().sort_values(by="stars", ascending=False)
df_gby_business[0:20].plot(kind="bar", x="business_id", figsize=(15, 7))
df_gby_business[-20:].plot(kind="bar", x="business_id", figsize=(15, 7))

# This graph not really can tell too much. Let's sum the stars and count the review id for each business and plot.
df_gby_business = df.groupby("business_id", as_index=False).agg({"review_id": "count", "stars": "mean"}).rename(
    columns={"stars": "star_mean", "review_id": "review_count"}).sort_values("review_count", ascending=False)
df_gby_business[0:20].plot(kind="bar", x="business_id", y=["review_count", "star_mean"], figsize=(15, 7)).axhline(y=4,
                                                                                                                  color='r',
                                                                                                                  linestyle='-')
df_gby_business[-20:].plot(kind="bar", x="business_id", y=["review_count", "star_mean"], figsize=(15, 7)).axhline(y=4,
                                                                                                                  color='r',
                                                                                                                  linestyle='-')

df_gby_business.describe()

'''
During data exploration, we found out number of reviews for each business is not very high.
The top one has 79 reviews in total. However, this dataset we only consider the first 10,000.
This also tells us the reviews are not balance in each business. Different business has different
number of reviews, which is very normal.  If we have detail for each business, we can group the business
by type and make a balance dataset. Such as group all Japanese food restaurants and generate a model. 
Based on the graph, the most popular business are has rating above 3.
'''

'''
Exploring word frequencies
'''

# Create a dataset slice, which only contain star = 5 and top business
top_business_id = df_gby_business.iloc[0]["business_id"]
top_business_text_df = df[(df["stars"] == 5) & (df["business_id"] == top_business_id)]
len(top_business_text_df)


# There are total 39 records

# Next, create a token for each corpus and merge all tokens in a single list


def my_tokenizer(text):
    return text.split() if text != None else []


top_business_tokens = top_business_text_df.text.map(my_tokenizer).sum()

# Now, we can count the frequencies
top_business_counter = Counter(top_business_tokens)
top_business_counter.most_common(20)

# let's remove all stop words.
from spacy.lang.en.stop_words import STOP_WORDS


def remove_stopwords(tokens):
    tokens = [token.lower() for token in tokens]
    return [t for t in tokens if t not in STOP_WORDS]


top_business_counter = Counter(remove_stopwords(top_business_tokens))
top_business_counter.most_common(20)

# Convert tuples to data frame
top_business_freq = pd.DataFrame.from_records(top_business_counter.most_common(20), columns=['token','count'])

# Create bar plot
top_business_freq.plot.bar(x="token")

'''
According to the graph, we can see the most frequency word is 'great', which makes sense because we reselct the
dataset is rating is 5. This tells us if the review text contains 'great', it is very high percentage which is 
belong to a high rating and they are happy with the service.
So that, we could based on the 'star' to classify to Positive or Negative.
'''

# Even though, we have some sort of conclusion, but let's use wordcloud to make the plot more visible and easier to understand
from wordcloud import WordCloud


def wordcloud(counter):
    wc = WordCloud(width=1200, height=800, background_color='white', max_words=200)
    wc.generate_from_frequencies(counter)

    #plot
    fig = plt.figure(figsize=(6,4))
    plt.imshow(wc, interpolation="none")
    plt.axis("off")
    plt.tight_layout(pdf=0)
    plt.show()

wordcloud(top_business_counter)

# Last but not least. let's check na values
print(df.isnull().values.sum())

"""
There are not NA value in this data-set. 
Based on our data exploration, we knew we can use 'stars' column to classify whether review is Positive or Negative.
Let's create a new column to store our Class/Label value,
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
STEP 3 - Clean data / Text pre-processing
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
    balance_data_count = 2300
    n_df = filter_df.loc[filter_df["class"] == 0][:balance_data_count]
    # number of negative rows

    print("Number of negative should be " + balance_data_count + ". Actual is ", len(n_df.loc[n_df["class"] == 0]))

    print("Number of positive should be 0. Actual is ", len(n_df.loc[n_df["class"] == 1]))

    p_df = filter_df.loc[filter_df["class"] == 1][:balance_data_count]
    # number of positive rows

    print("Number of positive should be " + balance_data_count + ". Actual is ", len(p_df.loc[p_df["class"] == 1]))

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
STEP 4 - Feature extraction from text
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
STEP 5 - Build Models
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
