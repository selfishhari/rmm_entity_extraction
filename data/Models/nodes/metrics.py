import logging
from typing import Any, Dict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
from datetime import datetime
import os





###################### All Metrics ######################

def _get_classwise_corrects(cm_df):
    """
    Takes confusion matrix and returns true positives in each class
    """
    
    corrects = [cm_df.iloc[i,i] for i in range(cm_df.shape[0])]
    
    return corrects

def get_cm_df(actuals, predicted, known_categories=None) -> pd.DataFrame:
    
    """
    Builds confusion matrix with classwise precision/recall
    """
    
    cm = confusion_matrix(actuals, predicted)
    
    if known_categories != None:
        
        cm_df = pd.DataFrame(cm, index=known_categories, columns=known_categories)
        
    else:
        
        cm_df = pd.DataFrame(cm)
        
        known_categories = cm_df.columns.tolist()
        
    corrects = _get_classwise_corrects(cm_df)
    
    cm_df["actuals"] = cm_df.sum(axis=1)
    
    cm_df.loc["predicted", :] = cm_df.sum(axis=0)
    
    cm_df["recall"] = corrects + [np.nan]
    
    cm_df.loc["precision", :] = corrects + [np.nan, np.nan]      
    
    cm_df["recall"] = cm_df["recall"] / cm_df["actuals"]
    
    cm_df.loc["precision", :] = cm_df.loc["precision", :] / cm_df.loc["predicted", :]    
    
    return cm_df[ ["recall"] + ["actuals"] + known_categories]


def get_all_error_metrics(actuals, predicted, known_categories):
    """
    Returns several error metrics
    """
    
    report_dict = {}
    
    report_dict["accuracy"] = accuracy_score(actuals, predicted)
    
    report_dict["precision"] = precision_score(actuals, predicted, average="micro")
    
    report_dict["recall"] = recall_score(actuals, predicted, average="micro")
    
    cm_df = get_cm_df(actuals, predicted, known_categories)

    report_dict["cm"] = cm_df
    
    print("accuracy", report_dict["accuracy"])
    
    return report_dict





###################### Logging #############################

def log_model_outputs(parameters: Dict, metrics_test:Dict, metrics_train:Dict, time_stamp):
    """
    creates and writes model output-confusion matrices
    """
    
    
    
    current_log = pd.DataFrame({
                  "time_stamp":[time_stamp], 
                  "model_parameters":[parameters], 
                  "test_accuracy":[metrics_test["accuracy"]], 
                  "test_precision":[metrics_test["precision"]], 
                  "test_recall":[metrics_test["recall"]],
                  "train_accuracy":[metrics_train["accuracy"]], 
                  "train_precision":[metrics_train["precision"]], 
                  "train_recall":[metrics_train["recall"]]
                 }
                
                )
    name = parameters["name"]
    
    cm_file_name = os.path.join("/home/viky/Desktop/freelancer/NB-Custom/logs/"+ name + "_"+ time_stamp + "_confusion_matrix.xlsx")
    
    cm_train = metrics_train["cm"]
    
    cm_test = metrics_test["cm"]
    
    
    writer = pd.ExcelWriter(cm_file_name, engine = 'xlsxwriter')
    
    cm_train.to_excel(writer, sheet_name="train")
    
    cm_test.to_excel(writer, sheet_name="test")
    
    writer.save()
    
    writer.close()
    
    return [current_log, cm_train, cm_test]
    
    
    
