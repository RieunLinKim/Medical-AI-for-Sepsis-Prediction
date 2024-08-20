import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from tqdm import tqdm
import json
import torch
import h5py
import re

from config import *

path1 = "../data/training/"
path2 = "../data/training_setB/"
fnames1 = os.listdir(path1)
fnames2 = os.listdir(path2)
for fname in fnames1:
    match = re.search(r'\.([^.]*)$', fname)
    if not match or match.group(1) != 'psv':
        fnames1.remove(fname)

for fname in fnames2:
    match = re.search(r'\.([^.]*)$', fname)
    if not match or match.group(1) != 'psv':
        fnames1.remove(fname)
        
fnames1.sort()
fnames2.sort()

### all pids from 0 to 40335
### all tids from 0
def get_patient_by_id_original(idx):
    path   = path1   if idx < 20336 else path2
    fnames = fnames1 if idx < 20336 else fnames2
    idx    = idx     if idx < 20336 else idx-20336
    file_path = path + fnames[idx]
    match = re.search(r'\.([^.]*)$', file_path)
    if match and match.group(1) == 'psv':
        return pd.read_csv(file_path, sep='|')
    else:
        print('Error loading original data {}'.format(file_path))
        sys.exit()

def get_patient_by_id_imputed(idx):
    return pd.read_csv('../data/imputed/p'+str(idx).zfill(6)+'.csv')

def get_patient_by_id_normalized(idx):
    return pd.read_csv('../data/normalized/p'+str(idx).zfill(6)+'.csv')

def get_patient_by_id_standardized(idx):
    return pd.read_csv('../data/standardized/p'+str(idx).zfill(6)+'.csv')

def get_synthetic_patient_by_id(idx):
    return pd.read_csv('../data/synthetic/p{}.csv'.format(str(idx).zfill(6)))

def prepare_hdf5():
    with h5py.File('../data/patient_data.h5', 'w') as f:
        for pid in tqdm(range(40336)):
            p = get_patient_by_id_standardized(pid)[All_COLS]
            grp = f.create_group(f'patient_{pid}')
            grp.create_dataset('data', data=p.to_numpy(), compression='gzip')

def get_patient_data(pid, start, end):
    with h5py.File('../data/patient_data.h5', 'r') as f:
        data = f[f'patient_{pid}/data'][start:end+1]
    return data.tolist()

def get_synthetic_patient_data(pid, start, end):
    with h5py.File('../data/synthetic_patient_data.h5', 'r') as f:
        data = f[f'patient_{pid}/data'][start:end+1]
    return data.tolist()

def plot(patient):
    sepsis = 1 in patient.SepsisLabel.unique()
    if sepsis:
        predict_hour = patient.SepsisLabel.ne(0).idxmax()
        sepsis_hour = predict_hour + 6
        print("Pateint developed sepis at hour ", sepsis_hour)
    else:
        print("Patient did not develop sepsis")
    patient = patient.drop(columns=['Unit1','Unit2','Gender','Age', 'HospAdmTime', 'SepsisLabel'])
    fig, axs = plt.subplots(len(patient.columns.values)-1, 1, sharex=True, figsize=(10,100))
    for i, col in enumerate(patient.columns.values[:-1]):
        patient.plot(x=['ICULOS'], y=[col], kind='scatter', ax=axs[i])
        plt.sca(axs[i])
        if sepsis:
            plt.axvline(x=predict_hour, color='b')
            plt.axvline(x=sepsis_hour, color='r')
    return

def print_confusion_matrix(y_label, y_pred):
    cm = confusion_matrix(y_label, y_pred)
    print("Confusion Matrix:")
    print(cm)

def plot_curves(rid, y_true, y_pred_prob):
    fpr, tpr, roc_thres = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, prc_thres = precision_recall_curve(y_true, y_pred_prob)
    prc_auc = average_precision_score(y_true, y_pred_prob)
    
    # Best cutoff according to ROC
    roc_thres[0] -= 1
    distances = np.sqrt((1-tpr)**2 + fpr**2)
    best_threshold_roc = roc_thres[np.argmin(distances)]

    # Plotting ROC and PRC
    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Annotate ROC Curve
    roc_indices = np.round(np.linspace(0, len(roc_thres) - 1, 10)).astype(int)
    for i in roc_indices:
        plt.annotate(f'{roc_thres[i]:.4f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(10,-10))

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % prc_auc)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Annotate PRC
    prc_indices = np.round(np.linspace(0, len(prc_thres) - 1, 10)).astype(int)
    for i in prc_indices[:-1]:  # Last point (recall=1) might not have a corresponding threshold
        plt.annotate(f'{prc_thres[i]:.4f}', (recall[i], precision[i]), textcoords="offset points", xytext=(10,10))
        
    plt.tight_layout()
    if not os.path.exists('../plots'):
        os.mkdir('../plots')
    plt.savefig('../plots/{}.png'.format(rid), dpi=300)
    plt.show()
    return best_threshold_roc

def save_pred(results, cutoff, rid):
    pure_fn = 0
    pure_fp = 0
    pure_tp = 0
    pure_tn = 0
    
    y_pred_trim = []
    y_pred = []
    y_label = []
    dir_path = '../results/'+rid

    total_patient = len(results)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for pid, result in tqdm(results.items()):
        result = list(dict(sorted(result.items())).values())
        p = get_patient_by_id_original(pid)
        if len(p) != len(result):
            result = [0]*(len(p)-len(result))+result
        df = pd.DataFrame(result, columns=['PredictedProbability'])
        df['PredictedLabel'] = df.apply(lambda row: 1 if row['PredictedProbability']>cutoff else 0, axis=1)
        filename = dir_path+'/p'+str(pid).zfill(6)+'.psv'
        df.to_csv(filename, mode='w+', index=False, header=True, sep='|')

        patient_pred = df['PredictedLabel'].tolist()
        patient_label = p['SepsisLabel'].tolist()

        y_pred.extend(patient_pred)
        y_label.extend(patient_label)

        if 1 in patient_label:
            if patient_pred == patient_label:
                pure_tp += 1
            elif not 1 in patient_pred:
                pure_fn += 1
        else:
            if patient_pred == patient_label:
                pure_tn += 1
            elif 1 in patient_pred:
                pure_fp += 1

    miss_classified = total_patient - pure_tp - pure_fn - pure_tn - pure_fp
    print(f"TP {pure_tp} FN {pure_fn} TN {pure_tn} FP {pure_fp}")
    print(f"Miss classified {miss_classified}")
        
    return y_label, y_pred, dir_path
        

def evaluate_model(model, runid, test_loader):
    results = {}
    y_prob = []
    y_label = []
    method = model.method
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            pid, rid, x_batch, y_batch, u_batch, mask_batch, _ = batch
            if method == 'ResNet':
                x_batch = x_batch.unsqueeze(1)
            if method == 'Transformer':
                x_batch = x_batch.to(device)
                mask_batch = mask_batch.to(device)
                outputs = model(x_batch, mask=mask_batch).tolist()
            else:
                x_batch = x_batch.to(device)
                outputs = model(x_batch).tolist()
            y_label.extend(y_batch.tolist())
            y_prob.extend(torch.sigmoid(outputs))
            for i, (p, r) in enumerate(zip(pid, rid)):
                if not p.item() in results:
                    results[p.item()] = {}
                results[p.item()][r.item()] = outputs[i] if isinstance(outputs[i], float) else outputs[i][0] 

    filename = '../results/{}_probs.json'.format(runid)
    with open(filename, 'w') as f:
        json.dump(results, f)

    return results, y_label, y_prob

def plot_trainset_curves(model, train_loader, rid):
    model.eval()
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader):
            x_batch, y_batch = batch
            y_batch = y_batch.unsqueeze(1)
            x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')
            outputs = model(x_batch)
            y_true.extend(y_batch.tolist())
            y_prob.extend(outputs.tolist())
    
    plot_curves(rid+'_train', y_true, y_prob)

