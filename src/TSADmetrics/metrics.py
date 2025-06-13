import numpy as np
import torch
from sklearn.metrics import auc
from torch import Tensor
from numpy.typing import NDArray
from typing import Sequence, Union, Tuple

"""
Evaluation metrics for anomaly detection.

1. Point-Adjusted Score (window-base) (DualTF [1])
    point_adjusted_score_window

2. Point-Adjusted Score (point-base) (Donut [2])
    point_adjusted_score_point

3. Point-Adjusted Score with Delay (point-base) (SR-CNN [3])
    point_adjusted_score_delay

4. Point-wise Score with Tolerance (DualTF)
    point_wise_margin

5. Point-wise Score (Normal)
    point_wise_score

6. Range-AUC and Range-VUS [4]
    RangeAUC, RangeVUS

References:
- [1] Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection, 
      Youngeun Nam, Susik Yoon, Yooju Shin, Minyoung Bae, Hwanjun Song, Jae-Gil Lee, and Byung SukLee.
- [2] Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications,
      Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, 
      Dan Pei, Yang Feng, Jie Chen, Zhaogang Wang, and Honglin Qiao.
- [3] Time-Series Anomaly Detection Service at Microsoft,
      Hansheng Ren, Bixiong Xu, Yujing Wang, Chao Yi, Congrui Huang, Xiaoyu Kou, Tony Xing, Mao Yang, Jie Tong, and Qi Zhang.
- [4] Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection,
      John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay, Aaron Elmore, and Michael J. Franklin.
"""

ArrayLike = Union[Sequence[Union[int, float]], NDArray[Union[np.integer, np.floating]], Tensor]
def _to_numpy(*arrays: ArrayLike) -> Tuple[NDArray[np.float64], ...]:
    """
    Convert `anom` and `score` to NumPy ndarrays of dtype float,
    regardless of whether they were list, ndarray, or torch.Tensor.
    """
    def _convert(x: ArrayLike) -> NDArray[np.float64]:
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy().astype(float)
        return np.asarray(x, dtype=float)
    return tuple(_convert(arr) for arr in arrays)

def _enumerate_thresholds(rec_errors, n=1000):
    # maximum value of the anomaly score for all time steps in the validation data
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    thresholds.append(th)

    print(f'Threshold Range: ({np.min(rec_errors)}, {np.max(rec_errors)}) with Step Size: {step_size}')

    for i in range(n-1):
        th = th + step_size
        thresholds.append(th)
    
    return thresholds

def _create_sequences(values, window_size, stride, historical=False):
    seq = []
    if historical:
        for i in range(window_size, len(values) + 1, stride):
            seq.append(values[i-window_size:i])
    else:
        for i in range(0, len(values) - window_size + 1, stride):
            seq.append(values[i : i + window_size])
   
    return np.stack(seq)



# --------------------------------- #
#   1. Point-Adjusted Score (DualTF)
#      Window-based evaluation
# --------------------------------- #

def point_adjusted_score_window(anom, label, thresholds=None, n=1000, margin=10):
    """
    Calculate the point-adjusted anomaly score.
    Details in the DualTF paper
    """
    print('\n ##### Point Adjusted Evaluation #####')   
    if thresholds is None:
        thresholds = _enumerate_thresholds(anom, n=n)
    
    anom, label = _to_numpy(anom, label)

    anom_seq = _create_sequences(anom, window_size=margin, stride=1)
    labels_seq = _create_sequences(label, window_size=margin, stride=1)
    
    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []

    for th in thresholds: # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0

        for t in range(len(anom_seq)): # for each sequence
            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(labels_seq[t] == 1)[0]), set(np.where(anom_seq[t] > th)[0])

            #if len(pred_anomalies) > 0 and len(pred_anomalies.intersection(true_anomalies)) > 0:
            if len(pred_anomalies) > 0 and len(true_anomalies) > 0:
                # correct prediction (at least partial overlap with true anomalies)
                TP_t = TP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) == 0:
                # correct rejection, no predicted anomaly on no true labels
                TN_t = TN_t + 1 
            elif len(pred_anomalies) > 0 and len(true_anomalies) == 0:
                # false alarm (i.e., predict anomalies on no true labels)
                FP_t = FP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) > 0:
                # predict no anomaly when there is at least one true anomaly within the seq.
                FN_t = FN_t + 1
        
        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
    
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))
    
    highest_th_idx = np.argmax(f1)
    print(f'Best Threshold: {thresholds[highest_th_idx]}')
    print("Best Precision : {:0.4f}, Best Recall : {:0.4f}, Best F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "best_precision": precision[highest_th_idx],
        "best_recall": recall[highest_th_idx],
        "best_f1": f1[highest_th_idx],
        'best_threshold': thresholds[highest_th_idx],
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds
    }




# --------------------------------- #
#   2. Point-Adjusted Score (Donut)
#      Point-based evaluation
# --------------------------------- #

def _point_adjust(anom, label, threshold=0.5):
    predict = (anom > threshold).astype(int)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = (label[0] == 1)
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos : sp]:
                new_predict[pos : sp] = 1
            else:
                new_predict[pos : sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos : sp]:
            new_predict[pos : sp] = 1
        else:
            new_predict[pos : sp] = 0

    return new_predict

def point_adjusted_score_point(anom, label, thresholds=None, n=1000):
    print('\n ##### Point Adjusted Score with Delay Evaluation #####')
    if thresholds is None:
        thresholds = _enumerate_thresholds(anom, n=n)
    
    anom, label = _to_numpy(anom, label)
    
    TP, TN, FP, FN = [], [], [], []
    precision, recall, fpr, f1 = [], [], [], []

    for th in thresholds:  # for each threshold
        pred = _point_adjust(anom, label, threshold=th)
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0

        for i in range(len(pred)):
            if pred[i] == 1 and label[i] == 1:
                TP_t += 1
            elif pred[i] == 0 and label[i] == 0:
                TN_t += 1
            elif pred[i] == 1 and label[i] == 0:
                FP_t += 1
            elif pred[i] == 0 and label[i] == 1:
                FN_t += 1
        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
    
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))
    
    highest_th_idx = np.argmax(f1)
    print(f'Best Threshold: {thresholds[highest_th_idx]}')
    print("Best Precision : {:0.4f}, Best Recall : {:0.4f}, Best F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "best_precision": precision[highest_th_idx],
        "best_recall": recall[highest_th_idx],
        "best_f1": f1[highest_th_idx],
        'best_threshold': thresholds[highest_th_idx],
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds
    }




# ----------------------------------------- #
#  3. Point-Adjusted score with delay (SR-CNN)
#     Point-based evaluation
# ----------------------------------------- #

def _point_adjust_delay(anom, label, threshold=0.5, delay=7):
    predict = (anom > threshold).astype(int)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = (label[0] == 1)
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

def point_adjusted_score_delay(anom, label, thresholds=None, n=1000, delay=7):
    print('\n ##### Point Adjusted Score with Delay Evaluation #####')
    if thresholds is None:
        thresholds = _enumerate_thresholds(anom, n=n)

    anom, label = _to_numpy(anom, label)
    
    TP, TN, FP, FN = [], [], [], []
    precision, recall, fpr, f1 = [], [], [], []

    for th in thresholds:  # for each threshold
        pred = _point_adjust_delay(anom, label, threshold=th, delay=delay)
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0

        for i in range(len(pred)):
            if pred[i] == 1 and label[i] == 1:
                TP_t += 1
            elif pred[i] == 0 and label[i] == 0:
                TN_t += 1
            elif pred[i] == 1 and label[i] == 0:
                FP_t += 1
            elif pred[i] == 0 and label[i] == 1:
                FN_t += 1
        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
    
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))
    
    highest_th_idx = np.argmax(f1)
    print(f'Best Threshold: {thresholds[highest_th_idx]}')
    print("Best Precision : {:0.4f}, Best Recall : {:0.4f}, Best F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "best_precision": precision[highest_th_idx],
        "best_recall": recall[highest_th_idx],
        "best_f1": f1[highest_th_idx],
        'best_threshold': thresholds[highest_th_idx],
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds
    }




# ----------------------------------------- #
#  4. Point-wise Score with tolerance (DualTF)
#     Point-based evaluation
# ----------------------------------------- #

def point_wise_margin(anom, label, thresholds=None, margin=5, n=1000):
    print('\n ##### Point-wise Score with Tolerance Evaluation #####')
    if thresholds is None:
        thresholds = _enumerate_thresholds(anom, n=n)
    
    anom, label = _to_numpy(anom, label)
    
    TP, TN, FP, FN = [], [], [], []
    precision, recall, fpr, f1 = [], [], [], []
    
    for th in thresholds:  # for each threshold
        pred = (anom > th).astype(int)
        true_idx = np.where(label == 1)[0]

        TP_t = FP_t = TN_t = FN_t = 0
        for i in range(len(pred)):
            within_tol = np.any(abs(true_idx - i) <= margin)
            if pred[i] == 1:                    # 予測が陽性
                if within_tol:                  # かつ tol 内に真異常
                    TP_t += 1                   # → TP
                else:
                    FP_t += 1                   # → FP
            else:                               # 予測が陰性
                if label[i] == 1:                  # しかし、真異常
                    FN_t += 1                   # → FN
                else:
                    TN_t += 1

        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
    
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))
    
    highest_th_idx = np.argmax(f1)
    print(f'Best Threshold: {thresholds[highest_th_idx]}')
    print("Best Precision : {:0.4f}, Best Recall : {:0.4f}, Best F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "best_precision": precision[highest_th_idx],
        "best_recall": recall[highest_th_idx],
        "best_f1": f1[highest_th_idx],
        'best_threshold': thresholds[highest_th_idx],
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds
    }



# ---------------------------------- #
#   5. Point-wise Score (Normal)
#      Point-based evaluation
# ---------------------------------- #

def point_wise_score(anom, label, thresholds=None, n=1000):
    print('\n ##### Point-wise Score #####')
    if thresholds is None:
        thresholds = _enumerate_thresholds(anom, n=n)
    
    anom, label = _to_numpy(anom, label)
    
    TP, TN, FP, FN = [], [], [], []
    precision, recall, fpr, f1 = [], [], [], []

    for th in thresholds:  # for each threshold
        pred = (anom > th).astype(int)
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0

        for i in range(len(pred)):
            if pred[i] == 1 and label[i] == 1:
                TP_t += 1
            elif pred[i] == 0 and label[i] == 0:
                TN_t += 1
            elif pred[i] == 1 and label[i] == 0:
                FP_t += 1
            elif pred[i] == 0 and label[i] == 1:
                FN_t += 1
        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)

    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))
    
    highest_th_idx = np.argmax(f1)
    print(f'Best Threshold: {thresholds[highest_th_idx]}')
    print("Best Precision : {:0.4f}, Best Recall : {:0.4f}, Best F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "best_precision": precision[highest_th_idx],
        "best_recall": recall[highest_th_idx],
        "best_f1": f1[highest_th_idx],
        'best_threshold': thresholds[highest_th_idx],
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds
    }




# ---------------------------------- #
#  6. Range-AUC and Range-VUS
# ---------------------------------- #

def _get_anomaly_range(label):
    '''
    Get anomaly range.

    Parameter
    ---------
    label: array-like, 1D numpy array or list
        arrays of binary values (anomaly labels, 0 or 1)

    Returns
    -------
    L: List[Tuple[int, int]]
        list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    L = []
    i = 0
    j = 0 

    while j < len(label):
        # print(i)
        while label[i] == 0:
            i+=1
            if i >= len(label):
                break
        j = i+1
        # print('j'+str(j))
        if j >= len(label):
            if j==len(label):
                L.append((i,j-1))

            break
        while label[j] != 0:
            j+=1
            if j >= len(label):
                L.append((i,j-1))
                break
        if j >= len(label):
            break
        L.append((i, j-1))
        i = j

    return L


def _extend_postive_range(x, window=5):
    """
    Extend the positive range of the anomaly label.
    Eq. (13) in the paper.

    Parameters
    ----------
    x: array-like, 1D numpy array or list
        arrays of binary values (anomaly labels, 0 or 1)
    window: int, optional
        the size of the window to extend the positive range, default is 5
    
    Returns
    -------
    label: numpy array
        the extended anomaly label
    """
    label = x.copy().astype(float)
    L = _get_anomaly_range(label)   # index of non-zero segments
    length = len(label)

    for k in range(len(L)):
        s = L[k][0] 
        e = L[k][1] 
        
        x1 = np.arange(e,min(e+window//2,length))
        label[x1] += np.sqrt(1 - (x1-e)/(window))
        
        x2 = np.arange(max(s-window//2,0),s)
        label[x2] += np.sqrt(1 - (s-x2)/(window))
        
    label = np.minimum(np.ones(length), label)
    return label


def _extend_postive_range_individual(x, percentage=0.2):
    label = x.copy().astype(float)
    L = _get_anomaly_range(label)   # index of non-zero segments
    length = len(label)

    for k in range(len(L)):
        s = L[k][0] 
        e = L[k][1] 
        
        l0 = int((e-s+1)*percentage)
        
        x1 = np.arange(e,min(e+l0,length))
        label[x1] += np.sqrt(1 - (x1-e)/(2*l0))
        
        x2 = np.arange(max(s-l0,0),s)
        label[x2] += np.sqrt(1 - (s-x2)/(2*l0))
        
    label = np.minimum(np.ones(length), label)
    return label


def TPR_FPR_RangeAUC(labels, pred, P, L):
    """
    Parameters
    ----------
    labels: numpy array
        binary anomaly labels (0 or 1)
    pred: numpy array
        binary predictions (0 or 1)
    P: int
        number of positive labels (sum of labels)
    L: List[Tuple[int, int]]
        list of ordered pairs representing the segments of positive labels
    """
    product = labels * pred
    TP = np.sum(product) # Eq. (14-1)
    FP = np.sum(pred) - TP # Eq. (14-2)
    
    P_new = (P + np.sum(labels)) / 2  # Eq. (15-1)
    N_new = len(labels) - P_new # Eq. (15-2)

    existence = 0
    for seg in L:
        if np.sum(product[seg[0]:(seg[1]+1)]) > 0:
            existence += 1
    existence_ratio = existence / len(L) # ExistanceR() in Eq. (16-1)

    TPR_RangeAUC = min(TP / P_new, 1) * existence_ratio # Eq. (16-1)
    
    FPR_RangeAUC = FP / N_new # Eq. (16-2)
    
    Precision_RangeAUC = TP / np.sum(pred) # Eq. (16-3)
    
    return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC


def RangeAUC(score, labels, window=10, percentage=None, n=1000, verbose=False):
    score, labels = _to_numpy(score, labels)
    score_sorted = -np.sort(-score)

    P = np.sum(labels)

    if percentage:
        labels = _extend_postive_range_individual(labels, percentage=percentage)
    else:
        labels = _extend_postive_range(labels, window=window)
    
    L = _get_anomaly_range(labels)

    TPR_list = [0]
    FPR_list = [0]
    Precision_list = [1]
    
    for i in np.linspace(0, len(score)-1, n).astype(int):
        th = score_sorted[i]
        pred = (score >= th).astype(int)
        TPR, FPR, Precision = TPR_FPR_RangeAUC(labels, pred, P, L)
        
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        Precision_list.append(Precision)
        
    TPR_list.append(1)
    FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
    
    tpr = np.array(TPR_list)
    fpr = np.array(FPR_list)
    prec = np.array(Precision_list)
    
    width = fpr[1:] - fpr[:-1]
    height = (tpr[1:] + tpr[:-1]) / 2
    Range_AUC_ROC = np.sum(width * height) # Eq. (10) in the paper
    
    width_PR = tpr[1:-1] - tpr[:-2]
    height_PR = (prec[1:] + prec[:-1]) / 2
    Range_AUC_PR = np.sum(width_PR * height_PR) # Eq. (11) in the paper

    f1 = []
    for i in range(len(prec)):
        f1.append(2 * (prec[i] * tpr[i]) / (prec[i] + tpr[i] + 1e-7))
    
    highest_th_idx = np.argmax(f1)
    if verbose:
        print(f'Best Threshold: {score_sorted[highest_th_idx]}')
        print("Best Precision : {:0.4f}, Best Recall : {:0.4f}, Best F-score : {:0.4f} ".format(
                prec[highest_th_idx], tpr[highest_th_idx], f1[highest_th_idx]))
    
    return {
        "range_auc": Range_AUC_ROC, 
        "range_pr": Range_AUC_PR, 
        "fpr": fpr, 
        "tpr": tpr, 
        "precision": prec,
        "best_precision": prec[highest_th_idx],
        "best_recall": tpr[highest_th_idx],
        "best_f1": f1[highest_th_idx],
        "best_threshold": score_sorted[highest_th_idx],
    }


def RangeVUS(score, labels_original, min_w=1, max_w=20, n=1000):
    tpr_3d=[]
    fpr_3d=[]
    prec_3d=[]
    
    auc_3d=[]
    ap_3d=[]
    
    window_3d = np.arange(min_w, max_w + 1, 1)

    score, labels_original = _to_numpy(score, labels_original)
    
    for window in window_3d:
        r_auc = RangeAUC(score, labels_original, window=window, n=n)
        
        tpr_3d.append(r_auc["tpr"])
        fpr_3d.append(r_auc["fpr"])
        prec_3d.append(r_auc["precision"])
        auc_3d.append(r_auc["range_auc"])
        ap_3d.append(r_auc["range_pr"])
    
    return {
        "tpr_3d": tpr_3d,
        "fpr_3d": fpr_3d,
        "prec_3d": prec_3d,
        "window_3d": window_3d,
        "vus_roc": sum(auc_3d) / len(window_3d),
        "vus_pr": sum(ap_3d) / len(window_3d)
    }