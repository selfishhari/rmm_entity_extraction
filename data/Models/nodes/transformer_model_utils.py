"""
Contains helper functions for transformer models viz. BERT, XLNet etc
"""

import json

from tempfile import TemporaryDirectory
from webchat_classification.nodes import feature_generation, metrics, bilstm, reports
import numpy as np
import pandas as pd
import scrapbook as sb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils_nlp.common.timer import Timer
from src.webchat_classification.nodes.nlp_recipes.utils_nlp.dataset.multinli import load_pandas_df
from utils_nlp.models.transformers.sequence_classification import (
    Processor, SequenceClassifier)





def get_data_loaders(train_data, test_data, parameters):
    
    BATCH_SIZE = parameters["batch_size"]

    NUM_GPUS = parameters["num_gpus"]

    MAX_LEN = parameters["max_len"]

    LABEL_COL = parameters["label_col"]

    LABEL_COL_ENC = parameters["label_col_enc"]

    TEXT_COL = parameters["text_col"]
    
    CACHE_DIR = TemporaryDirectory().name

    label_encoder = LabelEncoder()
    
    train_data[TEXT_COL] = train_data[TEXT_COL].astype("str")

    test_data[TEXT_COL] = test_data[TEXT_COL].astype("str")

    train_data[LABEL_COL_ENC] = label_encoder.fit_transform(train_data[LABEL_COL])

    test_data[LABEL_COL_ENC] = label_encoder.transform(test_data[LABEL_COL])

    num_labels = len(np.unique(train_data[LABEL_COL_ENC]))
    
    print("Number of unique labels: {}".format(num_labels))
    
    print("Number of training examples: {}".format(train_data.shape[0]))
    
    print("Number of testing examples: {}".format(test_data.shape[0]))
    
    processor = Processor(
        model_name=parameters["name"],
        to_lower=parameters["name"].endswith("uncased"),
        cache_dir=CACHE_DIR,
    )
    
    
    train_dataloader = processor.create_dataloader_from_df(
            train_data, TEXT_COL, LABEL_COL_ENC, max_len=MAX_LEN, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=True
        )

    test_dataloader = processor.create_dataloader_from_df(
            test_data, TEXT_COL, LABEL_COL_ENC, max_len=MAX_LEN, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=False
        )
    
    train_dataloader_noshuff = processor.create_dataloader_from_df(
            train_data, TEXT_COL, LABEL_COL_ENC, max_len=MAX_LEN, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=False
        )
    
    return train_dataloader, test_dataloader, train_dataloader_noshuff, train_data, test_data, label_encoder, CACHE_DIR


def train(train_dataloader, cache_dir, parameters):
    
    
    
    model = SequenceClassifier(
        model_name=parameters["name"], num_labels=parameters["num_labels"], cache_dir=cache_dir
    )
    
    model.fit(
            train_dataloader,
            num_epochs=parameters["epochs"],
            num_gpus=parameters["num_gpus"],
            verbose=parameters["verbose"],
            weight_decay=parameters["weight_decay"]
        )
    
    return model

def predictions(model, data_loader, data, label_encoder, parameters):
    # predict
    preds_probs = model.predict(
            data_loader, num_gpus=parameters["num_gpus"], verbose=parameters["verbose"], return_max_prob=True
        )

    data["pred_label_prob"] = preds_probs[1]

    data["pred_label_enc"] = preds_probs[0]

    data["pred_label"] = label_encoder.inverse_transform(data["pred_label_enc"])

    return data
