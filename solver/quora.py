# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
import string


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


def clean_up(text):
    # remove punctuation
    text = "".join(filter(lambda x: x not in string.punctuation, text))
    # tokenize text
    text = tokenize(text)
    # remove stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return [ps.stem(word) for word in text if word not in stop_words]


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

train.columns = ['id', 'question', 'target']

# truncate data
train = train[0:10]

train['tokenizer'] = train['question'].apply(tokenize)

# apply Counter Vectorizer
count_vector = CountVectorizer(analyzer=clean_up)
X_counts = count_vector.fit_transform(train['question'])
print(count_vector.get_feature_names())


# add new features
train['len'] = train['question'].apply(lambda x: len(x) - x.count(" "))
