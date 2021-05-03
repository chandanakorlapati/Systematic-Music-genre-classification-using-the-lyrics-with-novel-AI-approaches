'''
Genre Prediction Project

Pre processing the data. Including 2 phases:
Phase1: Clean the dataset,remove non-english songs,  remove invalid genres, remove duplicated lyrics...
Phase2: Pre process lyrics, remove stop words, lemmatization, change to lower 
form 

'''

# import necessary libraries
import sys
import os
import zipfile

# csv file
import csv
import pandas as pd
import numpy as np

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


# df: input dataframe
def pre_process_1(input_df):
    df = input_df
    # Different genres of dataset
    genre_lst = df['genre'].unique()
    
    # Simply remove songs with 'Not Available' genre and 'Other' genre
    rm_genre_lst = ['Not Available','Other']
    genre_lst = [i for i in genre_lst if i not in rm_genre_lst]
    df = df[~df['genre'].isin(rm_genre_lst)]
    
    # Simply remove song with empty lyrics
    df = df[df['lyrics'].notna()]
    
    # Just remove all duplicated lyrics
    same_lysic = df['lyrics'].value_counts()
    df = df[~df['lyrics'].isin(same_lysic[same_lysic>=2].index)]
    
    # Remove all lyrics with less than 200 (about ~ 40 words)
    df = df[df['lyrics'].str.len() > 200]

    # Detect if the song is english and removing
    ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS
    no_eng_cnt = 0
    non_eng_df = pd.DataFrame()
    rm_index = []
    for index, row in df.iterrows():
        text = row.lyrics.lower()
        words = set(nltk.wordpunct_tokenize(text))
        if len(words & ENGLISH_STOPWORDS) <= len(words & NON_ENGLISH_STOPWORDS):
            no_eng_cnt += 1
            rm_index.append(index)

    df = df.drop(rm_index)
    return df


# input: input dataframe
# return a new processed dataframe
def pre_process_2(input_df):
    df = input_df
    # Set the lyrics after modify back to df
    for index, row in df.iterrows():
        tmp_lyric = [word.lower() for word in nltk.word_tokenize(row['lyrics']) if word.isalpha()]
        tmp_lyric = ' '.join([lemmatizer.lemmatize(word) for word in tmp_lyric if word not in eng_stopwords])
        df['lyrics'][index] = tmp_lyric
    
    return df


# This main function simple performs the first phase of of pre processing
# And save to the path that you provided
# Modify the code if you want to do the second phase (could take time)
def main(argv):
    if len(argv) != 3:
        print('usage: python input_csv_file output_csv_file')
        sys.exit()
    df = pd.read_csv(argv[1], index_col=0, encoding='utf8')
    df_preprocess_phase1 = pre_process_1(df)
    df_preprocess_phase1.to_csv(argv[2])

if __name__ == '__main__':
    main(sys.argv)
