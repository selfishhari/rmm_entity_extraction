""" Data Preparation"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import string





###################  Cleaning Data ############################

def _clean_text(x, cleaning_params):
    
    """
    Basic cleaning function to clean texts.
    
    Cleaning params is used to control what is cleaned. 
    Custom characters can also be provided, which when provided, will solely be used to be remove chars.
    
    Eg:
    cleaning_params = {"puncts":True,
               "digits":True,
               "custom":False, #if True, custom_chars_to_remove will be used to remove characters
               "custom_chars_to_remove":"",
               "remove_system_messages":True}
               
    """
    
    x = str(x)
    
    x = x.lower()
    
    puncts = string.punctuation
    
    digits = string.digits
    
    chars_to_remove = ""
    
    if cleaning_params["puncts"]:
        
        chars_to_remove = chars_to_remove + puncts
        
    if cleaning_params["digits"]:
        
        chars_to_remove = chars_to_remove + digits
        
    if cleaning_params["custom"]:
        
        chars_to_remove = cleaning_params["custom_chars_to_remove"]
        
    translator = str.maketrans('', '', chars_to_remove)
    
    x = x.translate(translator)
    
    return x

def _collate_labels(data:pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Merge labels as mentioned in parameters
    """
    
    merge_mapping = params["merge_labels"]
    
    top_classes = params["top_classes"]
    
    map_dict = {}
    
    for new_label in merge_mapping:
        
        for old_label in merge_mapping[new_label]:
            
            map_dict[old_label] = new_label
            
    data = data.replace( {"label":map_dict})
    
    all_new_labels = data["label"].unique()
    
    map_dict = {}
    
    if params["clean_params"]["collate_label"]:


        for old_class in all_new_labels:

            if (old_class in top_classes) | (old_class == "") | pd.isnull(old_class):

                continue

            else:

                if "others" != old_class:

                    map_dict[old_class] = "others"

        data = data.replace( {"label":map_dict})
    
    return data

def clean_text_and_labels(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Clean the raw text and normalize the labels.
    """
    
    cleaning_params = params["clean_params"]
    
    data["clean_text"] = data["filtered_message"].apply(lambda x: _clean_text(x, cleaning_params))
    
    data["label"] = data["label"].str.lower()
    
    if cleaning_params["merge_labels"]:
        
        data = _collate_labels(data, params)
        

    return data





###################  Filtering Data ############################

def _remove_ambiguous_convs(data: pd.DataFrame) -> pd.DataFrame:
    """
    removes sessions which have been marked as ambiguous.
    """
    
    ambi_session_ids = data.loc[~pd.isnull(data["ambiguous_flag"]), "session_id"].tolist()
    
    data = data.loc[~data.session_id.isin(ambi_session_ids), :]
    
    return data

def _filter_till_labeled(data: pd.DataFrame, data_lables: pd.DataFrame) -> pd.DataFrame:
    """
    Filters conversations only till that message where label has been provided.
    (This is done to select only those conversations till which point, humans(annotaters) were able to identify the intent)
    """
    
    data_lables = data_lables.copy()
    
    data_lables.columns = data_lables.rename({"message_n":"cut_message_n"}, axis=1).columns # Renaming the message numbers column to have the message_n at which data was labelled.
    
    data = data[["name_anon", "session_id", "message_n", "clean_text", "ambiguous_flag"]].merge(data_lables[["session_id", "cut_message_n"]], on="session_id", how="left")
    
    data = data.loc[data.message_n <= data.cut_message_n, :].reset_index(drop=True).copy()
    
    return data


def filter_data(data:pd.DataFrame, filter_params: Dict) -> pd.DataFrame:    
    """
    Remove ambiguous cases and select only required chats
    """
    
    data_labels = data.loc[~pd.isnull(data.label), :].copy() #select only those chats where labels are present
    
    data_labels = data_labels.drop_duplicates(subset="session_id") # Remove multiple labels in a session_id. Should ideally not change anything unless the labelled data has multiple labels per session_id in which case only the first label will be considered.
    
    if filter_params["remove_ambiguous"]:
        
        data = _remove_ambiguous_convs(data)
        
    if filter_params["filter_till_labeled"]:
        
        data = _filter_till_labeled(data, data_labels)
        
    if filter_params["remove_system_messages"]:
        
        data = data.loc[data.name_anon != "system", :].reset_index(drop=True)
        
    # merge the processed data with labels
        
    data = data[["name_anon", "session_id", "message_n", "clean_text"]].merge(data_labels[["session_id", "label", "ambiguous_flag"]],
                                                                              on="session_id", how="left")
    
    return data





###################  Collating Data ############################

def _collate_texts(x):
    """
    helper function to merge texts within a session_id
    """
    
    x = x.dropna()
    
    return " ".join(x.values)

def collate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge texts in a single session_id to form 1 row per session_id
    """
    
    data_labels = data.loc[~pd.isnull(data.label), :].copy() #select only those chats where labels are present
    
    data_labels = data_labels.drop_duplicates(subset="session_id") #        
    
    collated_data = data.groupby("session_id")["clean_text"].apply(_collate_texts).reset_index()
    
    collated_data = collated_data.merge(data_labels[["session_id", "label"]], on="session_id", how="left")
    
    collated_data = collated_data.dropna()
    
    collated_data = collated_data.loc[collated_data["clean_text"] != "", :]
    
    return collated_data


def line_classifer_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Just take the line where the label was given
    """
    
    data_labels = data.loc[~pd.isnull(data.label), :].copy() #select only those chats where labels are present
    
    data_labels = data_labels.drop_duplicates(subset="session_id") #        
    
    data_labels = data_labels.loc[data_labels["clean_text"] != "", :]
    
    return data_labels





def collate_labels_pred(data:pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Merge labels as mentioned in parameters for predictions in dataframe
    """
    
    merge_mapping = params["merge_labels"]
    
    top_classes = params["top_classes"]
    
    map_dict = {}
    
    for new_label in merge_mapping:
        
        for old_label in merge_mapping[new_label]:
            
            map_dict[old_label] = new_label
            
    data = data.replace( {"pred_label":map_dict})

    data = data.replace( {"label":map_dict})
    
    all_new_labels = data["label"].unique()

    print(all_new_labels)
    
    map_dict = {}
    
    for old_class in all_new_labels:
        
        if (old_class in top_classes) | (old_class == "") | pd.isnull(old_class):
            
            continue
            
        else:
            
            if "others" != old_class:
                
                map_dict[old_class] = "others"
            
    data = data.replace( {"pred_label":map_dict})

    data = data.replace( {"label":map_dict})
    
    return data
