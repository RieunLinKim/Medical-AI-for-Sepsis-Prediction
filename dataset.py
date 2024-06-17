import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import re

from utils import get_patient_by_id_original, get_patient_by_id_standardized, get_patient_data, get_synthetic_patient_by_id
from config import padding_offset

DATA_FOLDER = "../data"
processing = {
    "original"    : "/",
    "imputed"     : "/imputed3/",
    "normalized"  : "/normalized3/",
    "standardized": "/standardized/",
    "standardized_padded": "/standardized_padded/"
}

COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
       'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
       'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
       'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'ICULOS']

class SepsisDataset(Dataset):
    def __init__(self, patient_ids, seq_len=72, starting_offset=24, cols=COLS, method='standardized_padded'):
        self.patient_ids = patient_ids
        self.method = method
        self.seq_len = seq_len
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(seq_len, starting_offset)
        
    def check_store(self, seq_len, starting_offset):
        try:
            directory = '../data/idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'_(\d+)'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == starting_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self, seq_len, starting_offset):
        index_map_subset = self.check_store(seq_len, starting_offset)
        if not index_map_subset:
            path = '../data/idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(40336), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_original(pid)
                    if len(p) < starting_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'SepsisLabel'])
                        hist = (pid,0,t,label,seq_len-t-1) # patient id, start, end, label, padding
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[4] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(starting_offset-1,len(p)):
                            label = int(p.at[t, 'SepsisLabel'])
                            hist = (pid,0,t,label, seq_len-t-1) if t < seq_len else (pid,t-seq_len+1,t,label,0)
                            assert hist[2] < len(p)
                            assert hist[2]-hist[1]+1+hist[4] == seq_len
                            index_map.append(hist)
                            patients_all.add(pid)
                            if pid in self.patient_ids:
                                index_map_subset.append(hist)
                                self.ratio[label] += 1
                                patients_subset.add(pid)
                print('populated {} patients into {} timeseries'.format(len(patients_all), len(index_map)))
                with open(path, "w") as fp:
                    json.dump(index_map, fp)
            else:
                with open(path, "r") as fp:
                    index_map = json.load(fp)
                    for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=75):
                        pid = item[0]
                        label = item[-1]
                        if pid in self.patient_ids:
                            index_map_subset.append(item)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('using {} patients in current subset'.format(len(patients_subset)))

            assert len(patients_subset) == len(self.patient_ids)
            assert patients_subset == set(self.patient_ids)
            print('len idxmap {}'.format(len(index_map)))
            path = '../data/idxmap_subset/idxmap_subset_'+str(seq_len)+'_'+str(starting_offset)+'_'+str(len(index_map_subset))+".json"
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids, index_map_subset=index_map_subset), fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, padding = self.idxmap_subset[idx]
        data = [0]*padding + get_patient_data(pid, start, end)
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(mask), torch.tensor([])


class RawDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        self.x = []
        self.y = []
        self.ids = []
        for pid in tqdm(pids, desc="Preparing data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            self.x.extend(p[COLS].values.tolist())
            self.y.extend(p['SepsisLabel'].tolist())
            self.ids.extend([(pid, rid) for rid in range(len(p))])
        print('Populated {} dps from {} patients'.format(len(self.y), len(pids)))
        return
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ids[idx][0], self.ids[idx][1], torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor([]), torch.tensor([])


class SyntheticDataset(Dataset):
    def __init__(self, pids, seq_len=72, starting_offset=24):
        self.seq_len = seq_len
        self.patient_ids = pids
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(seq_len, starting_offset)


    def check_store(self, seq_len, starting_offset):
        try:
            directory = '../data/idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'syn_[^0-9]*(_\d+)+'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == starting_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self, seq_len, starting_offset):
        index_map_subset = self.check_store(seq_len, starting_offset)
        if not index_map_subset:
            path = '../data/syn_idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(10000), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_original(pid)
                    if len(p) < starting_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'SepsisLabel'])
                        hist = (pid,0,t,label,seq_len-t-1) # patient id, start, end, label, padding
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[4] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(starting_offset-1,len(p)):
                            label = int(p.at[t, 'SepsisLabel'])
                            hist = (pid,0,t,label, seq_len-t-1) if t < seq_len else (pid,t-seq_len+1,t,label,0)
                            assert hist[2] < len(p)
                            assert hist[2]-hist[1]+1+hist[4] == seq_len
                            index_map.append(hist)
                            patients_all.add(pid)
                            if pid in self.patient_ids:
                                index_map_subset.append(hist)
                                self.ratio[label] += 1
                                patients_subset.add(pid)
                print('populated {} patients into {} timeseries'.format(len(patients_all), len(index_map)))
                with open(path, "w") as fp:
                    json.dump(index_map, fp)
            else:
                with open(path, "r") as fp:
                    index_map = json.load(fp)
                    for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=75):
                        pid = item[0]
                        label = item[-1]
                        if pid in self.patient_ids:
                            index_map_subset.append(item)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('using {} patients in current subset'.format(len(patients_subset)))

            assert len(patients_subset) == len(self.patient_ids)
            assert patients_subset == set(self.patient_ids)
            print('len idxmap {}'.format(len(index_map)))
            path = '../data/idxmap_subset/syn_idxmap_subset_'+str(seq_len)+'_'+str(starting_offset)+'_'+str(len(index_map_subset))+".json"
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids, index_map_subset=index_map_subset), fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, padding = self.idxmap_subset[idx]
        data = [0]*padding + get_patient_data(pid, start, end)
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(mask), torch.tensor([])


class WeibullCoxDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        path = '../data/wc_dataset.json'
        with open(path, 'r') as f:
            try:
                data = json.load(f)
                if set(data['pids']) == set(pids):
                    self.x = data['x']
                    self.tau = data['tau']
                    self.S = data['S']
                    self.ids = data['ids']
                    self.y = data['y']
                    return
            except Exception as err:
                print('Error matching saved Weibull-Cox dataset: ', err)
        
        self.x = []
        self.tau = []
        self.S = []
        self.ids = []
        self.y = []
        for pid in tqdm(pids, desc="Preparing data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            for rid in range(len(p)-5):
                window = p.iloc[rid:min(len(p),rid+6),:]
                S = 0 if (window['SepsisLabel'] == 1).any() else 1
                tau = max(0.1, window['SepsisLabel'].idxmax()-rid if (window['SepsisLabel'] == 1).any() else 7)
                x = p.loc[rid, COLS].tolist()
                y = p.loc[rid, ['SepsisLabel']].tolist()
                self.x.append(x)
                self.y.append(y)
                self.tau.append(tau)
                self.S.append(S)
                self.ids.append((pid, rid))

        data = {
            'pids': pids,
            'x'   : self.x,
            'tau' : self.tau,
            'S'   : self.S,
            'y'   : self.y,
            'ids' : self.ids
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return
        
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, idx):
        return self.ids[idx][0], self.ids[idx][1], torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.tau[idx], dtype=torch.float32), torch.tensor(self.S[idx], dtype=torch.int32)