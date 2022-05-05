import numpy as np
import torch
from typing import List

def accuracy_score(y_label: np.ndarray, y_pred_label: np.ndarray):
    assert len(y_pred_label) == len(y_label), "Prediction and groundtruth not have the same length."
    corrected = np.sum(y_label == y_pred_label).astype(np.float64)
    return 1.0 * corrected / len(y_label)

def precision_score(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=1):
    score = -1
    if average == 'binary':
        tp = np.sum((y_label == pos_label) * (y_pred_label == pos_label))
        fp = np.sum((y_label == 1 - pos_label) * (y_pred_label == pos_label))
        score = 1.0 * tp / (tp + fp)
    elif average == 'micro':
        tp = np.sum(y_label == y_pred_label)
        fp = np.sum(y_label != y_pred_label)
        score = 1.0 * tp / (tp + fp)

    else:   # 'macro'
        score_0 = precision_score(y_label, y_pred_label, pos_label=0)
        score_1 = precision_score(y_label, y_pred_label, pos_label=1)
        score = (score_0 + score_1) / 2
    return score

def recall_score(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=0):
    score = -1
    if average == 'binary':
        tp = np.sum((y_label == pos_label) * (y_pred_label == pos_label))
        fn = np.sum((y_label == pos_label) * (y_pred_label == 1 - pos_label))
        score = 1.0 * tp / (tp + fn)
    elif average == 'micro':
        tp = np.sum(y_label == y_pred_label)
        fn = np.sum(y_label != y_pred_label)
        score = 1.0 * tp / (tp + fn)

    else:   # 'macro'
        score_0 = recall_score(y_label, y_pred_label, pos_label=0)
        score_1 = recall_score(y_label, y_pred_label, pos_label=1)
        score = (score_0 + score_1) / 2
    return score

def f1_score(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=0):
    score = -1
    if average == 'binary':
        precision = precision_score(y_label, y_pred_label, average='binary', pos_label=pos_label)
        rec = recall_score(y_label, y_pred_label, average='binary', pos_label=pos_label)
        score = 2.0 * precision * rec / (precision + rec)
    elif average == 'micro':
        precision = precision_score(y_label, y_pred_label, average='micro')
        rec = recall_score(y_label, y_pred_label, average='micro')
        score = 2.0 * precision * rec / (precision + rec)
    else:
        f1_0 = f1_score(y_label, y_pred_label, average='binary', pos_label=0)
        f1_1 = f1_score(y_label, y_pred_label, average='binary', pos_label=1)
        score = (f1_0 + f1_1)/2
    return score

def calculate_cls_metric(y_label: np.ndarray, y_pred_label: np.ndarray):
    return 

def calculate_cls_metrics(y_label: np.ndarray, y_pred_label: np.ndarray):
    accuracy = accuracy_score(y_label, y_pred_label)
    mic_pre, mic_rec, mic_f1 = calculate_cls_metric(y_label, y_pred_label, average='micro')
    mac_pre, mac_rec, mac_f1 = calculate_cls_metric(y_label, y_pred_label, average='macro')
    real_pre, real_rec, real_f1 = calculate_cls_metric(y_label, y_pred_label, average='binary', pos_label=0)
    fake_pre, fake_rec, fake_f1 = calculate_cls_metric(y_label, y_pred_label, average='binary', pos_label=1)
    return accuracy, (real_pre, real_rec, real_f1),\
                     (fake_pre, fake_rec, fake_f1),\
                     (mic_pre, mic_rec, mic_f1),\
                     (mac_pre, mac_rec, mac_f1)
                     
if __name__ == '__main__':
    print(1)
    # f1_score(np.array([]), np.array([0.]))