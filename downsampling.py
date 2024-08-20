import json
from utils import get_patient_by_id_original
from tqdm import tqdm
import random

# with open("../data/sepsis_ids.json", "r") as f:
#     sepsis_ids = json.load(f)
#     print(len(sepsis_ids))

# with open("../data/train_pos.json", "r") as f:
#     train_ids = json.load(f)

# inspecting sepsis patients
sepsis_ids = []
for i in tqdm(range(40336), desc="Inspecting sepsis", ascii=False, ncols=75):
    patient = get_patient_by_id_original(i)
    if 1 in patient["SepsisLabel"].unique():
        sepsis_ids.append(i)

print("sepsis ids: ", len(sepsis_ids))

with open("../data/sepsis_ids.json", "w") as f:
    json.dump(sepsis_ids, f)

with open("../data/train_ids.json", "r") as f:
    train_ids = json.load(f)

train_pos = [value for value in train_ids if value in sepsis_ids]
print("train_pos: ", len(train_pos))
print("train_ids: ", len(train_ids))

with open("../data/train_pos.json", "w") as f:
    json.dump(train_pos, f)

# Randomly choose 50 from train_pos, if train_pos has more than 50 elements
if len(train_pos) > 50:
    train_pos_random = random.sample(train_pos, 50)

with open("../data/train_pos_random.json", "w") as f:
    json.dump(train_pos_random, f)

print("train_pos_random: ", len(train_pos_random))