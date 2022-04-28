# Nigeria-News-Detective

[Streamlit Application Link](https://share.streamlit.io/bombaybrownboy09/nigeria-news-detective/main/app.py)

**Final Project by Bhargav Shetgaonkar for Duke AIPI 540**
<p align="center"><img align="center" width="800px" src="images/NN1.jpeg"></p>

<a name="proj-stat"></a>
## 1. Problem statement
The objective of this project is to train a deep learning model to classify Nigeria News articles on violent events based on sources and predict entities linked to ones mentioned in the text using NLP

<a name="proj-struc"></a>
## 2. Project Structure
The project data and codes are arranged in the following manner:

```
├── README.md               <- description of the project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── datasets                <- contains the ACLED data from Jan 2019 to Apr 2022
├── models                  <- contains models of label encoding, word2vec, logreg and entity prediction model (Sequential Neural Net)
├── .gitignore              <- exploratory and main .ipynb files used
├── setup                   <- setup files for both models
├── app.py                  <- app to run project / user interface
├── .gitignore              <- git ignore file

```

_Data_: <br>
the `data` folder can be downloaded from below link:
1) Download data [here](https://github.com/BombayBrownBoy09/Nigeria-News-Detective/tree/main/datasets) 
    - **Training data:** Sourced from ACLED data on Nigeria from Jan 2019 to Apr 2022
    -  **Validation data:** you can set a 0.20 validation split while training
2) Download trained models [here](https://github.com/BombayBrownBoy09/Nigeria-News-Detective/tree/main/models)

```sh
https://github.com/schmidtdominik/RecipeNet/raw/master/simplified-recipes-1M.npz
```

<a name="exp"></a>
## 3. Experimentation
We have 2 DL models working in this application:

**Model1 (Sentence Transformer Text Classification):**
The best performing news source classification model used a pretrained [sentence transformer](https://github.com/BombayBrownBoy09/Nigeria-News-Detective/blob/main/notebooks/Text_Classification_Sentence_transformer.ipynb) model to generate embeddings and then used a logistic regression classification to obtain results (20 % train acc). The other approach tried was [bag of words](https://github.com/BombayBrownBoy09/Nigeria-News-Detective/blob/main/notebooks/Text_Classification_Bag_of_Words.ipynb) which resulted in lesser 11.2% train acc
<!-- <p align="center"><img align="center" width="800px" src="data/Word2Vec.png"></p>
<p align="center"><img align="center" width="800px" src="data/Word2Vec Acc + Loss.png"></p> -->
To get source of news, run the following from the home directory:

```sh
python setup/text_classification.py
```

**Model2 (Linked Entity Prediction using Word2vec embeddings and LSTM neural net):**
We first used NER using nltk and then got lists of entities for each event. We uniformly make list size to 3 and mask the 3rd entity to create a target variable Y. We then use word2vec to generate embeddings and generate prediction using a 4 layer sequential neural net [here](https://github.com/BombayBrownBoy09/Nigeria-News-Detective/blob/main/notebooks/Entity_prediction.ipynb)

To get linked entities of news, run the following from the home directory:

```sh
python setup/entity_prediction.py
```

To run the streamlit app

```sh
python setup/app.py
```

This will prompt a user input in the command line for a news text and output the top 5 linked entities along with news source.
<p align="center"><img align="center" width="800px" src="images/Screen Shot 2022-04-26 at 10.23.55 PM.png"></p>
