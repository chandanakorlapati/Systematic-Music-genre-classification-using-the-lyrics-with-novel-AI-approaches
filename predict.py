'''
    Genre Prediction Project
    Bao Nguyen - Teja - Nagasai
    
    This file perform predict genre of an input song (in text file)
    Basically, this load the a trained model, and do pre-process the input song then predict
    
'''

import sys
import numpy as np
# For export model
import joblib

# nltk
import nltk
# Download nltk package
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
eng_stopwords = set(stopwords.words('english'))
#lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# Preprocess a song
# Input: lyric of an input song
def preprocess_a_lyric(input):
  tmp_lyric = [word.lower() for word in nltk.word_tokenize(input) if word.isalpha()]
  tmp_lyric = ' '.join([lemmatizer.lemmatize(word) for word in tmp_lyric if word not in eng_stopwords])
  return tmp_lyric


# Predict genre a song base on lyrics
# input: lyrics of an input song after pre-process, clf: loaded model to predict
def predict_a_song(input, clf):
  #load_model_dir = 'Model/genrePredict.pkl'
  #clf = joblib.load(load_model_dir) 
  out = preprocess_a_lyric(input)
  return clf.predict([out])[0]
  

# Simple do the prediction with a lyric of a song
def main(argv):
    if len(argv) != 2:
        print('usage python predict.py input')
        sys.exit()

    f = open(argv[1], "r")
    
    data = f.read()
    load_model_dir = 'Model/genrePredict.pkl'
    clf = joblib.load(load_model_dir)
    print("Genre of this lyric is ", predict_a_song(data, clf))


if __name__ == '__main__':
    main(sys.argv)
