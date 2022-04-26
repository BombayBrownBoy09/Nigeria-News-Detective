# -*- coding: utf-8 -*-
"""Text Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OuDH_50Ta_mpx0PL4iCo9NoRcrMsAJqq
"""

import os
import numpy as np
import pandas as pd
import string
import time
import urllib.request
import zipfile
import torch

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data1 = pd.read_csv(r'/content/Nigeria2019_ACLED_Extract.csv')
data2 = pd.read_csv(r'/content/2020-01-01-2022-04-01-Western_Africa-Nigeria.csv')
df = pd.concat([data1, data2], axis=0)

df = df[['notes', 'source']]
label_encoder = preprocessing.LabelEncoder()
df['source']= label_encoder.fit_transform(df['source'])
from sklearn.model_selection import train_test_split
X = df['notes']
y = df['source']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.tolist()
X_test = X_test.tolist()

# Load pre-trained model
senttrans_model = SentenceTransformer('all-MiniLM-L6-v2',device=device)

# Create embeddings for training set text
X_train = X_train
X_train = [senttrans_model.encode(doc) for doc in X_train]

# Create embeddings for test set text
X_test = X_test
X_test = [senttrans_model.encode(doc) for doc in X_test]

# Train a classification model using logistic regression classifier
y_train = pd.Series(y_train)
logreg_model = LogisticRegression(solver='saga')
logreg_model.fit(X_train,y_train)
preds = logreg_model.predict(X_train)
acc = sum(preds==y_train)/len(y_train)
print('Accuracy on the training set is {:.3f}'.format(acc))

# Evaluate performance on the test set
y_test = pd.Series(y_test)
preds = logreg_model.predict(X_test)
acc = sum(preds==y_test)/len(y_test)
print('Accuracy on the test set is {:.3f}'.format(acc))

import joblib
file1 = 'logreg_model.joblib'
joblib.dump(logreg_model, file1)

file2 = 'label_encoder.joblib'
joblib.dump(label_encoder, file2)

file3 = 'senttrans_model.joblib'
joblib.dump(senttrans_model, file3)