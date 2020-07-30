""" Generate feature for various models """

import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from stop_words import get_stop_words
from nodes import glove
import os

from importlib import reload
reload(glove)

from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing.sequence import pad_sequences

########### Split Data #############
def split_data(data: pd.DataFrame, params: Dict):
    """
    Split collated data into train/test
    """
    
    nrows = data.shape[0]
    
    train_len = int(nrows * (1 - params["test_data_ratio"]))
    
    test_len = nrows - train_len
    
    train_data = data.sample( train_len, random_state=params["random_seed"]).copy()
    
    train_idxs = train_data.index.tolist()
    
    test_data = data.loc[~data.index.isin(train_idxs), :].copy()
    
    return [train_data, test_data]


def split_data_one_v_all(data: pd.DataFrame, params: Dict, params_class: Dict):
    """
    Split collated data into train/test with subsampling all other classes for one vs all model
    """
    
    class_data = data.loc[data["label"] == params_class["class_name"], :].copy()
    
    class_data["label"] = 1
    
    class_nrows = class_data.shape[0]
    
    non_class_len = params["subsample_factor_ova"] * class_nrows
    
    nrows = non_class_len + class_nrows
    
    all_non_class_data = data.loc[data["label"] != params_class["class_name"], :]
    
    subsampled_non_class_data = all_non_class_data.sample(non_class_len, random_state=params["random_seed"]).copy()
    
    subsampled_non_class_data["label"] = 0
    
    subsampled_data = pd.concat( [class_data, subsampled_non_class_data], axis=0, ignore_index = True)
    
    train_len = int(nrows * (1 - params["test_data_ratio"]))
    
    test_len = nrows - train_len
    
    train_data = subsampled_data.sample( train_len, random_state=params["random_seed"]).copy()
    
    train_idxs = train_data.index.tolist()
    
    test_data = subsampled_data.loc[~subsampled_data.index.isin(train_idxs), :].copy()
    
    # Building data for reporting
    
    class_ratio = (class_nrows / (data.shape[0] - class_nrows ))
    
    test_data_label_size = test_data.loc[test_data["label"] == 1, :].shape[0]                   
    
    train_ids = train_data["session_id"].tolist()
    
    test_ids = test_data["session_id"].tolist()
    
    non_class_data_rep = all_non_class_data.loc[~all_non_class_data["session_id"].isin(train_ids + test_ids), :].copy()
    
    rep_sample = int( (1/class_ratio) * test_data_label_size) - (test_data.shape[0] - test_data_label_size)
    
    rep_non_class_data_rep = non_class_data_rep.sample(rep_sample, random_state=params["random_seed"]).copy()
    
    rep_non_class_data_rep["label"] = 0
    
    rep_data = pd.concat( [test_data, rep_non_class_data_rep], axis=0, ignore_index = True)
    
    #rep_data = data.copy()
    
    #rep_data.loc[rep_data["label"] == params_class["class_name"],"label"] = 1 
    
    #rep_data.loc[rep_data["label"] != 1,"label"] = 0    
    
    return [train_data, test_data, rep_data, class_ratio]



def split_data_one_v_all_v2(data: pd.DataFrame, params: Dict, class_name: str):
    """
    Split collated data into train/test with subsampling all other classes for one vs all model
    """
    data_split, rep_data = split_data(data, params)
    
    data_split_y, known_categories = _encode_target_var(data_split["label"])
    
    rep_data["label_enc"], known_categories = _encode_target_var(rep_data["label"], known_categories)
    
    class_data = data_split.loc[data["label"] == class_name, :].copy()
    
    class_data["label"] = 1
    
    class_nrows = class_data.shape[0]
    
    non_class_len = params["subsample_factor_ova"] * class_nrows
    
    nrows = non_class_len + class_nrows
    
    all_non_class_data = data_split.loc[data_split["label"] != class_name, :]
    
    if non_class_len > all_non_class_data.shape[0]:
        
        subsampled_non_class_data = all_non_class_data.copy()
        
        nrows = all_non_class_data.shape[0] + class_nrows
        
    else:
        
        subsampled_non_class_data = all_non_class_data.sample(non_class_len, random_state=params["random_seed"]).copy()
    
    subsampled_non_class_data["label"] = 0
    
    subsampled_data = pd.concat( [class_data, subsampled_non_class_data], axis=0, ignore_index = True)
    
    train_len = int(nrows * (1 - params["test_data_ratio"]))
    
    test_len = nrows - train_len
    
    train_data = subsampled_data.sample( train_len, random_state=params["random_seed"]).copy()
    
    train_idxs = train_data.index.tolist()
    
    test_data = subsampled_data.loc[~subsampled_data.index.isin(train_idxs), :].copy()
    
    # Building data for reporting
    
    class_ratio = (class_nrows / (data.shape[0] - class_nrows ))
    
    return [train_data, test_data, rep_data, class_ratio, known_categories]
    
    #data_split.loc[data_split["label"] == class_name, "label"] = 1
    
    #data_split.loc[data_split["label"] != 1, "label"] = 0

    #return [data_split, rep_data, rep_data, class_ratio, known_categories]
    





########### Generate Ngram Features #############

def _encode_target_var(target_series: pd.Series, known_categories=None)->pd.DataFrame:
    
    if known_categories==None:
        
        factorized_series = target_series.factorize()
        
        known_categories = factorized_series[1].tolist()
        
        encoded_target = factorized_series[0]
    
    else:
        
        encoded_target = [known_categories.index(i) for i in target_series.tolist()]
        
    return [encoded_target, known_categories]

def _decode_target_var(target_array, known_categories: list)->pd.DataFrame:
    
    decoded_target = [known_categories[x] for x in target_array]
        
    return decoded_target

def generate_ngram_features(train_data: pd.DataFrame, test_data: pd.DataFrame, feature_gen_params: Dict) -> pd.DataFrame:
    
    """
    Generate ngram based features on test and train data    
    """
    
    if feature_gen_params["remove_stopwords"]:
        
        stopwords = get_stop_words("en")
        
    else:
        
        stopwords = None
        
    if feature_gen_params["tfidf"]:
        
        vectorizer = TfidfVectorizer(stop_words=stopwords, 
                                     ngram_range=(feature_gen_params["ngram_start"], feature_gen_params['ngram_end']), 
                                     max_features=feature_gen_params["max_features"])
        
    train_x = vectorizer.fit_transform(train_data["question "])
    
    test_x = vectorizer.transform(test_data["question "])
    
    train_y, known_categories = _encode_target_var(train_data["part"])
    
    test_y, known_categories = _encode_target_var(test_data["part"], known_categories)
    
    test_data["label_enc"] = test_y
    
    train_data["label_enc"] = train_y
        
    return [train_x, train_y, test_x, test_y, known_categories, vectorizer, train_data, test_data]



def generate_ngram_features_ova(train_data: pd.DataFrame, test_data: pd.DataFrame, report_data: pd.DataFrame, feature_gen_params: Dict) -> pd.DataFrame:
    
    """
    Generate ngram based features on test and train data    
    """
    
    if feature_gen_params["remove_stopwords"]:
        
        stopwords = get_stop_words("en")
        
    else:
        
        stopwords = None
        
    if feature_gen_params["tfidf"]:
        
        vectorizer = TfidfVectorizer(stop_words=stopwords, 
                                     ngram_range=(feature_gen_params["ngram_start"], feature_gen_params['ngram_end']), 
                                     max_features=feature_gen_params["max_features"])
        
    train_x = vectorizer.fit_transform(train_data["clean_text"])
    
    test_x = vectorizer.transform(test_data["clean_text"])
    
    report_x = vectorizer.transform(report_data["clean_text"])
    
    train_y = train_data["label"]
    
    test_y = test_data["label"]
    
    report_y = report_data["label"]
    
    test_data["label_enc"] = test_y
    
    report_data["label_enc"] = report_y
    
    train_data["label_enc"] = train_y
        
    return [train_x, train_y, test_x, test_y, report_x, report_y, vectorizer, train_data, test_data, report_data]
        
    
def generate_glove_features(train_data: pd.DataFrame, test_data: pd.DataFrame, feature_gen_params: Dict) -> pd.DataFrame:
    
    """
    Convert to embeddings
    """
    
    vectorizer = glove.GloveVectorizer(
                emb_size = feature_gen_params["emb_size"], # 50, 100, 200, 300
                 method = feature_gen_params["method"], # "multilength", "average"
                 max_length = feature_gen_params["max_length"], #Maximum number of words to be encoded in a sentence. Is ignored when method="average"
                 remove_stop_words = feature_gen_params["remove_stopwords"],
                 embedding_filepath = os.path.join("data","01_raw", "glove.6B", "glove.6B." + str(feature_gen_params["emb_size"]) + "d.txt")
    )
        
    train_x = vectorizer.transform(train_data["clean_text"])
    
    test_x = vectorizer.transform(test_data["clean_text"])
    
    train_y, known_categories = _encode_target_var(train_data["label"])
    
    test_y, known_categories = _encode_target_var(test_data["label"], known_categories)
    
    test_data["label_enc"] = test_y
    
    train_data["label_enc"] = train_y
        
    return [train_x, train_y, test_x, test_y, known_categories, vectorizer, train_data, test_data]

def generate_sequences(train_data: pd.DataFrame, test_data: pd.DataFrame, feature_gen_params: Dict) -> pd.DataFrame:
    
    """
    Creates sequences of embeddings to be sent to a sequential dnn model
    """
    
    vectorizer = KerasTokenizer(num_words=feature_gen_params["vocab_size"])
    
    train_data["clean_text"] = train_data["clean_text"].apply(lambda x: str(x))
    
    test_data["clean_text"] = test_data["clean_text"].apply(lambda x: str(x))
    
    vectorizer.fit_on_texts(train_data["clean_text"])
    
    train_x = vectorizer.texts_to_sequences(train_data["clean_text"])
    
    test_x = vectorizer.texts_to_sequences(test_data["clean_text"])
    
    train_x = pad_sequences(train_x, maxlen=feature_gen_params["max_length"])
    
    test_x = pad_sequences(test_x, maxlen=feature_gen_params["max_length"])
    
    train_y, known_categories = _encode_target_var(train_data["label"])
    
    test_y, known_categories = _encode_target_var(test_data["label"], known_categories)
    
    test_data["label_enc"] = test_y
    
    train_data["label_enc"] = train_y
    
    train_y = pd.get_dummies(train_y)
    
    test_y = pd.get_dummies(test_y)
        
    return [train_x, train_y, test_x, test_y, known_categories, vectorizer, train_data, test_data]
