"""
This script encodes text with glove embeddings

author: narahari.b
"""

import os
import numpy as np
import pandas as pd
from stop_words import get_stop_words


class GloveVectorizer(object):
    
    """
    Provide averaged embeddings for a sentence
    Provide multilength embeddings for a sentence
    Text preprocessing features
    """
    
    def __init__(self, 
                 emb_size = 50,
                 method = "multilength", # "multilength", "average"
                 max_length = 100, #Maximum number of words to be encoded in a sentence. Is ignored when method="average"
                 remove_stop_words = True,
                 vocab_size = 60000,
                 embedding_filepath = "../../data/01_raw/glove.6B/glove.6B.50d.txt"
                ):
        
        self.emb_size = emb_size
        
        self.method = method
        
        self.max_length = max_length
        
        self.remove_stop_words = remove_stop_words
        
        self.embedding_filepath = embedding_filepath
        
        self.embeddings = {}
        
        self.vocab_size = vocab_size
        
        
        return
    
    def load_word_embeddings(self):
        
        self.embeddings = {}
        
        with open(self.embedding_filepath,'r') as infile:
            
            for line in infile:
                
                values = line.split()
                
                self.embeddings[ values[0] ] = np.asarray( values[1:], dtype='float32')
                
        return self.embeddings
    
    
    def get_embeddings(self, s):
        
        # ignore stop words
        
        if self.remove_stop_words:
            
            stop_words = get_stop_words("en")
            
        else:
            
            stop_words = [""]
            
        words = s.split()
        
        if self.method == "multilength":
            
            words = words[:self.max_length]
        
        text_embeddings = []
        
        for w in words:
            
            if (w.isalpha()) & (w in self.embeddings.keys() ) & (not (w in stop_words)):
                
                text_embeddings += [self.embeddings[w]]
                
            else:
                
                text_embeddings += [self.embeddings["#"]]
        
        if self.method == "average":
            
            """
            Takes sentence and creates embeddings resulting (embedding_size) features.
            Averages embeddings from all words
            """
            
            return np.array(text_embeddings).mean(axis = 0)
        
        else:
            """
            Takes sentence and creates embeddings resulting (max_length * embedding_size) features
            """
            
            return np.array(text_embeddings).flatten()
    
    
    def transform(self, text: pd.Series) -> pd.DataFrame:
        
        if len(self.embeddings.keys()) == 0:
            
            self.load_word_embeddings()
            
        embeddings = text.apply(lambda x: self.get_embeddings(str(x)))
        
        return pd.DataFrame(embeddings.values.tolist())
    
    def get_embedding_matrix(self, word_indxs):
        
        """
        Return embedding matrix in the order of word indexes
        """
        
        if len(self.embeddings.keys()) == 0:
            
            self.load_word_embeddings()
            
        reordered_embeddings = {}
        
        all_embeddings = np.stack(self.embeddings.values())
        
        all_words_embeddings = self.embeddings.keys()
        
        emb_mean = all_embeddings.mean()
        
        emb_std = all_embeddings.std()
        
        # Create default matrix of size numwords * emb_size with mean and std calculated above.
        # If any word is not available in our embeddings it will have a normal dist of mean and std
        
        embedding_matrix = np.random.normal(emb_mean, emb_std, (self.vocab_size, self.emb_size))
            
        for word, i in word_indxs.items():
            
            if i >= self.vocab_size:
                
                continue
            
            if word in all_words_embeddings:
                
                embedding_matrix[i] = self.embeddings[word]
                
                
        return embedding_matrix
                
    #def multilength_embeddings(self):
        
        
        
        
        