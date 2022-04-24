# making all essential imports
import nltk
import joblib
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import re
import torch
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.layers import Embedding
from sentence_transformers import SentenceTransformer
from tensorflow.keras.preprocessing.text import Tokenizer


# using gensim import word2vec and vectorize the recipes list
word2vec = joblib.load('word2vec.joblib')
entity_model = keras.models.load_model('entity_model.h5', compile=False)


def notes_to_entities(notes):
    words = nltk.word_tokenize(notes)
    tag = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(tag, binary=False)
    entities =[]
    labels =[]
    for chunk in chunks:
        if hasattr(chunk,'label'):
            entities.append(' '.join(c[0] for c in chunk))
            labels.append(chunk.label())
        
    entities = ','.join(entities)
    return entities

def get_linked_entities(input_news):
    text = notes_to_entities(input_news)
    input_text = text[:].split(',')
    input_text = input_text[0:2]

    # converts text to vector
    input_vector = [word2vec[idx] if idx in word2vec else np.zeros((100,)) for idx in input_text]
    if len(input_vector)!=2:
      while len(input_vector)!=2:
        input_vector.append(np.zeros((100,)))

    # converts input vector to numpy array
    input_vector = np.array([input_vector])

    # # getting model predictions for top 10 suggestions by finding the 10 most similar ingredients to our output
    output_vector = entity_model.predict(input_vector)
    pred = word2vec.most_similar(positive=[output_vector.reshape(100,)], topn=5)
    return pred


model = joblib.load('logreg_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
senttrans_model = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')

def source_find(input_news):
    X = senttrans_model.encode(input_news)
    pred = model.predict([X,])
    pred = label_encoder.inverse_transform(pred)
    return pred


def run():
    # Streamlit page title
    st.title("Source and Linked Entities of your news")
    st.markdown('**This is a demo application*')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    # Clear input form
    def clearform():
        st.session_state['newtext'] = ''

    # Input form
    with st.form('reviewtext'):
        new_review = st.text_area(label='Write or paste news text below',
                                    value = '',
                                    key='newtext')
        b1,b2 = st.columns([1,1])
        with b1:
            submit = st.form_submit_button(label='Submit')
        with b2:
            clear = st.form_submit_button(label='Reset', on_click=clearform)

    if submit and new_review !='':
        # Generate prediction
        source = source_find(new_review)
        entity = get_linked_entities(new_review)

        # Display the prediction
        st.markdown('### Predicted source: {}'.format(source[0]))
        # st.markdown('#### {}'.format(source[0]))
        st.markdown('### Predicted entities:')
        for i in range(len(entity)):
            st.markdown('#### {}. {} : {}'.format(i+1, entity[i][0], entity[i][1]))

if __name__ == "__main__":
    run()
