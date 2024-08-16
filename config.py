SEQ_LEN    = 120
SEQ_OFFSET = 48
BATCH_SIZE = 512
LR         = 0.001
EPOCHS     = 100
INPUT_DIM  = 37
HIDDEN_DIM = 64
FF_DIM     = 1024
N_CLASSES  = 2
N_HEADS    = 16
N_LAYERS   = 8

ORGAN_HIDDEN_DIM = 16
ORGAN_N_HEADS    = 4
ORGAN_N_LAYERS   = 4
ORGAN_FF_DIM     = 128

sepsis_ids_filepath = "../data/sepsis_ids.json"
train_ids_filepath = "../data/train_pos.json"
# train_ids_filepath = "../data/train_ids.json"   # split into trainng and testing, manually truncate some of it
test_ids_filepath = "../data/test_ids.json"
synthetic_train_ids_filepath = '../data/synthetic_train_ids.json'
synthetic_test_ids_filepath = '../data/synthetic_test_ids.json'
label_dirpath = '../results/labels/'
normals_filepath = "../data/normals.json"

wandb_project = 'sepsis-prediction'

preprocess_method = "standardized_padded"

All_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
       'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
       'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
       'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Unit1', 'Unit2', 'Gender',
       'HospAdmTime', 'ICULOS']

COL_IDX_MAP = {'HR': 0, 'O2Sat': 1, 'Temp': 2, 'SBP': 3, 'MAP': 4, 'DBP': 5, 'Resp': 6, 'EtCO2': 7, 'BaseExcess': 8, 'HCO3': 9, 'FiO2': 10, 'pH': 11, 'PaCO2': 12, 'SaO2': 13, 'AST': 14, 'BUN': 15, 'Alkalinephos': 16, 'Calcium': 17, 'Chloride': 18, 'Creatinine': 19, 'Bilirubin_direct': 20, 'Glucose': 21, 'Lactate': 22, 'Magnesium': 23, 'Phosphate': 24, 'Potassium': 25, 'Bilirubin_total': 26, 'TroponinI': 27, 'Hct': 28, 'Hgb': 29, 'PTT': 30, 'WBC': 31, 'Fibrinogen': 32, 'Platelets': 33, 'Age': 34, 'Unit1': 35, 'Unit2': 36, 'Gender': 37, 'HospAdmTime': 38, 'ICULOS': 39}


clinical_measurements_dict = {
    "Cardiovascular": ["HR", "SBP", "MAP", "DBP", "TroponinI"],
    "Respiratory": ["O2Sat", "Resp", "EtCO2", "FiO2", "PaCO2", "SaO2", "pH"],
    "Immune": ["Temp", "WBC"],
    "Coagulation": ["Platelets", "PTT", "Fibrinogen"],
    "Renal": ["BUN", "Creatinine", "BaseExcess", "HCO3", "pH", "Chloride", "Magnesium", "Phosphate", "Potassium"],
    "Liver": ["AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total"],
    "Hematologic": ["Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"],
    "Metabolism": ["BaseExcess", "HCO3", "pH", "Glucose", "Lactate", "Calcium"],
    "Demographic": ["Age", "Gender", "ICULOS"]
}


organ_system_ids = {
    'Cardiovascular': [0, 3, 4, 5, 27],
    'Respiratory': [1, 6, 7, 10, 12, 13, 11],
    'Immune': [2, 31],
    'Coagulation': [33, 30, 32],
    'Renal': [15, 19, 8, 9, 11, 18, 23, 24, 25],
    'Liver': [14, 16, 20, 26],
    'Hematologic': [28, 29, 30, 31, 32, 33],
    'Metabolism': [8, 9, 11, 21, 22, 17],
    "Demographic": [34, 35, 36]
}