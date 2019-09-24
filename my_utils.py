import csv
import numpy as np
import pandas as pd


def load_glove_embedding(file_name):
    print("Loading Glove Embedding")
    with open(file_name, 'r') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            which_word = line[0]
            words.add(which_word)
            word_to_vec[which_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        word_to_index = {}
        index_to_word = {}
        for w in sorted(words):
            word_to_index[w] = i
            index_to_word[i] = w
            i = i+1
    print("Done", len(word_to_vec), "words loaded!")
    return word_to_index, index_to_word, word_to_vec
    

def read_csv(filename = 'my_data/myTrainDataset.csv'):
    sentence = []
    sentence_type = []
    
    with open(filename) as csvMyData:
        csvReader = csv.reader(csvMyData)
        for row in csvReader:
            sentence.append(row[0])
            sentence_type.append(row[1])
    
    X = np.asarray(sentence)
    Y = np.asarray(sentence_type, dtype = int)
    
    return X, Y


def sentences_to_indices(X, word_to_index, max_len):
    """
    X -- array of sentences, of shape (N, 1)
    word_to_index -- a dictionary mapping word to index
    max_len -- maximumm number of words in a sentence
    
    returns:
    X_indices -- array of indices of the words in the sentence of shape (N, max_len)
    """
    N = X.shape[0]
    X_indices = np.zeros((N, max_len))
    
    for i in range(N):
        words = X[i].lower().split()
        j=0
        for w in words:
            X_indices[i,j] = word_to_index[w]
            j = j+1
   
    return X_indices



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    