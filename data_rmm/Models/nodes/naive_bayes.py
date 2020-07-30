"""

"""

from sklearn.naive_bayes import MultinomialNB
import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
from nodes import feature_generation
from importlib import reload
from datetime import datetime
import joblib
import os
reload(feature_generation)





######### train model ##############################

def train(train_x, train_y, test_x, test_y, parameters):
    
    model = MultinomialNB(alpha=0.2)
    
    model = model.fit(train_x.toarray(), train_y)
    
    return model





########### predictions #########################

def predictions(X, y, model, known_categories):
    
    preds_probs = model.predict_proba(X.toarray())

    print((X.toarray()).shape)
    preds_label = model.predict(X.toarray())
    
    preds_label_dec = feature_generation._decode_target_var(preds_label, known_categories)
    
    y["pred_label"] = preds_label_dec
    
    y["pred_label_enc"] = preds_label
    
    y["pred_label_prob"] = np.max(preds_probs, axis=1)
    
    return [preds_label, preds_label_dec, preds_probs, y]


"""
def predictions_ova(X, y, model):
    
    dmatrix = xgb.DMatrix(X, label = y["label_enc"])
    
    preds_probs = model.predict(dmatrix)

    y["pred_label_prob"] = preds_probs[:, 1]
    
    y["pred_label"] = (y["pred_label_prob"] > 0.5).astype(int)
    
    y["pred_label_enc"] = y["pred_label"]
    
    return [y["pred_label"].tolist(), y["pred_label"].tolist(), preds_probs, y]
    
"""




########### save model ##########################

def save(model, parameters):
    
    name = parameters["name"]
    
    time_stamp = datetime.now().strftime("%d_%b_%Y_%H_%M_%S_%f")
    print(os.path)
    
    joblib.dump(model, os.path.join("/home/viky/Desktop/freelancer/NB-Custom/models/"+ name + "_"+ time_stamp + ".model"))
    
    return time_stamp
