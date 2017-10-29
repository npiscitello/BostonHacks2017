import os
import json
import random
from shutil import copyfile


src = os.getcwd()
print("Warning: You must delete all data except raw for this to split the data correctly")


print("Splitting data into training and validation...")
# Split the data into 70% train data and 30% val data


#print("src: " + src)
directory = ["train/benign", "train/malignant", "val/benign", "val/malignant"]
for i in range(0, len(directory)):
    path = src + "/data/" + directory[i]
    if not os.path.exists(path):
        os.makedirs(path)

os.chdir(src + "/data/raw")
file_list = os.listdir()
json_list = []
for i in range(0, len(file_list) - 1):
    if file_list[i].endswith('.json'):
        json_list.append(file_list[i])

percentage_of_val_data = 0.30
num_val = int(len(json_list) * percentage_of_val_data)
while len(json_list) > num_val:
    rand = random.randint(0, len(json_list) - 1)
    file_data = json.load(open(json_list[rand]))
    del json_list[rand]
    file_name = file_data["name"] + ".jpg"
    #print(str(rand) + ": " + file_name)
    raw_path = src + "/data/raw/" + file_name
    if file_data["meta"]["clinical"]["benign_malignant"] == "benign":
        copyfile(raw_path, src + "/data/train/benign/" + file_name[5:])
    else:
        copyfile(raw_path, src + "/data/train/malignant/" + file_name[5:])
while len(json_list) > 0:
    rand = random.randint(0, len(json_list) - 1)
    file_data = json.load(open(json_list[rand]))
    del json_list[rand]
    file_name = file_data["name"] + ".jpg"
    raw_path = src + "/data/raw/" + file_name
    if file_data["meta"]["clinical"]["benign_malignant"] == "benign":
        copyfile(raw_path, src + "/data/val/benign/" + file_name[5:])
    else:
        copyfile(raw_path, src + "/data/val/malignant/" + file_name[5:])
