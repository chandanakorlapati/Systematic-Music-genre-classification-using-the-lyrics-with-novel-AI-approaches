'''
    Genre Prediction Project
    Bao Nguyen - Teja - Nagasai

    This files include various type of training
    (on tfidf of lyrics and can be on title)  including balancing the data

'''

# csv file
import sys
import numpy as np
import csv
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import confusion_matrix


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from sklearn.utils import resample

# For Neuron Network
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D

# For export model
import joblib


# For feature union
# When training with 2 features
# For feature union
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].values.astype('U')


# Getting train set, validation set
# Reserve a validation set With 400 songs for each genre
# Training set with 15000 songs for each genres,
# Do upsampling
# df_clean: input dataframe
# return validation set, training set
def get_training_set(df_clean):
    genre_lst = df_clean.genre.unique()
    genre_lst.sort()
    random_seed  = 100
    N = 400 # Size of each genre for test
    
    # Validation set
    val_df = pd.DataFrame()
    rest_df = pd.DataFrame()
    for genre in genre_lst:
        subset = df_clean[df_clean['genre'] == genre]
        val_set = subset.sample(n = N, random_state=random_seed)
        rest_set = subset.drop(val_set.index)
        rest_df = rest_df.append(rest_set)
        val_df = val_df.append(val_set)

    rest_df = shuffle(rest_df)
    val_df = shuffle(val_df)

    #Training set
    N = 15000 # Size of each genre for train
    train_df = pd.DataFrame()
    for genre in genre_lst:
        if rest_df[rest_df.genre == genre].shape[0] >= N:
            tmp = rest_df[rest_df.genre == genre].sample(n = N, random_state=27)
        else:
            tmp = resample(rest_df[rest_df.genre == genre],
                    replace=True, # sample with replacement
                    n_samples=N, # match number in majority class
                    random_state=27) # reproducible results
        train_df = train_df.append(tmp)

    train_df = shuffle(train_df)
    return (train_df, val_df)


# Train only on lyrics (tfidf)
# train_df: training set
# model: which model use naives bayer (0)  or logistic regression (1)

def train_lyrics_only(train_df, model):
    if(model == 0):
        text_clf = Pipeline([('vect', TfidfVectorizer()),
            ('clf', MultinomialNB(alpha=0.12))])
    else:
        text_clf = Pipeline([('vect', TfidfVectorizer()),
            ('clf', LogisticRegression(solver='lbfgs', 
                multi_class='multinomial', penalty='l2', C=0.6))])
    text_clf.fit(train_df.lyrics.values.astype('U'), train_df.genre) 
    return text_clf


# Train on lyrics and song titles (tfidf)
# train_df: training set
# model: which model use naives bayer (0)  or logistic regression (1)
def train_lyrics_and_titles(train_df, model):
    tmp = [MultinomialNB(), LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2', C=0.6)]
    text_clf = Pipeline([('feats', FeatureUnion([
                                ('Mlyrics', Pipeline([
                                    ('lyric-data',ItemSelector(key='lyrics')),
                                    ('tf-idf-lyrics', TfidfVectorizer())
                                             ])),
                                ('Mtitle', Pipeline([
                                    ('lyric-data',ItemSelector(key='song')),
                                    ('tf-idf-title', TfidfVectorizer())
                                             ]))])),
                     ('clf', tmp[model])])

    text_clf.fit(train_df, train_df.genre)
    return text_clf



# Training using Neuron Network
# train_df Training Set
# val_df Validation Sets
def train_NN(train_df, val_df):
    train_size = train_df.shape[0]
    test_size = val_df.shape[0]
    train_posts = train_df['lyrics']
    train_tags = train_df['genre']
    test_posts = val_df['lyrics']
    test_tags = val_df['genre']

    ### Process training data
    max_words = 10000 #number of top words to keep
    tokenize = text.Tokenizer(num_words=max_words, char_level=False) #num_words=max_words,

    tokenize.fit_on_texts(train_posts) # only fit on train
    x_train = tokenize.texts_to_matrix(texts = train_posts, mode='tfidf')
    x_test = tokenize.texts_to_matrix(texts = test_posts, mode='tfidf')

    # Use sklearn utility to convert label strings to numbered index
    encoder = LabelEncoder()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    # Converts the labels to a one-hot representation
    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)




    # This model trains very quickly and 2 epochs are already more than enough
    # Training for more epochs will likely lead to overfitting on this dataset
    # You can try tweaking these hyperparamaters when using this model with your own data
    batch_size = 100
    epochs = 2

    # Build the model
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    # model.fit trains the model
    # The validation_split param tells Keras what % of our training data should be used in the validation set
    # You can see the validation loss decreasing slowly when you run this
    # Because val_loss is no longer decreasing we stop training to prevent overfitting
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

    # Evaluate the accuracy of our trained model
    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# This main function simple performs the training for which model that you want
# Usage: python training.py input_df model_mode (0, 1, 2 (NN))
def main(argv):
    if len(argv) != 3:
        print('Usage: python training.py input_df model_mode (0, 1, 2 (NN))')
        sys.exit()
    df_clean = pd.read_csv(argv[1], index_col=0, encoding='utf8')
    df_clean = df_clean[df_clean['lyrics'].notna()]
    (train_df, val_df) = get_training_set(df_clean)
    print('train size', train_df.shape[0])
    print('test size', val_df.shape[0])
    if(argv[2] == '0'):
        text_clf = train_lyrics_only(train_df, 0)
        predicted = text_clf.predict(val_df.lyrics.values.astype('U'))
        print('Accuracy of MultinomialNB', np.mean(predicted == val_df.genre))
    if(argv[2] == '1'):
        text_clf = train_lyrics_only(train_df, 1)
        predicted = text_clf.predict(val_df.lyrics.values.astype('U'))
        print('Accuracy of logistic regression', np.mean(predicted == val_df.genre))

if __name__ == '__main__':
    main(sys.argv)

