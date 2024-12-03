import csv
import json
import random
from tqdm.auto import tqdm

data = []
label_list = ['DATE', 'CARDINAL', 'DECIMAL','MEASURE', 'MONEY', 'ORDINAL', 'TIME', 'DIGIT', 'FRACTION', 'TELEPHONE', 'ADDRESS']

label_group = { label: [] for label in label_list }
label_group["O"] = []

def read_data(csv_file):
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)

        # idx = 0
        tokens, tags, decodes, labels = [], [], [], []
        for row in tqdm(reader):
            # print(row)  

            token, tag, decode = row["Input Token"], row["Semiotic Class"], row["Output Token"]
            if tag not in label_list:
                tag = "O"
            if token == "<eos>":
                assert len(tokens) == len(tags)
                sample = {
                    "sentence": " ".join(tokens),
                    "tokens":tokens,
                    "tags":tags,
                    "decodes":decodes,
                    "labels":labels,
                }
                data.append(sample)
                if labels != []:
                    for l in labels:
                        label_group[l].append(sample)
                else:
                    label_group["O"].append(sample)
                tokens, tags, decodes, labels = [], [], [], []
            else:
                # if tag == "O":
                #     tokens.append(token)
                #     tags.append(tag)
                # else:
                #     if tag not in labels:
                #         labels.append(tag)
                    # sub_tokens = token.split()
                    # tokens.extend(sub_tokens)
                    # tags.extend([tag]*len(sub_tokens))
                    # tags.append(f"B-{tag}")
                    # tags.extend([f"I-{tag}"] * (len(sub_tokens)-1) )
                if tag not in labels and tag != "O":
                    labels.append(tag)
                tokens.append(token)
                tags.append(tag)
                decodes.append(decode)
            # idx += 1
            # if idx == 500000:
            #     break
# data_files = ["./dataset/output_1.csv"]
data_files = ["./dataset/output_1.csv", "./dataset/output_6.csv", "./dataset/output_11.csv",
              "./dataset/output_16.csv", "./dataset/output_21.csv","./dataset/output_91.csv",
              "./dataset/output_96.csv",]
for file in data_files:
    read_data(file)
    
print(len(data))

# with open("./data.json","w") as f:
#     json.dump(data,f,indent=2,ensure_ascii=False)
    
for k,v in label_group.items():
    print(f"{k}: {len(v)}")

print("-"*100)
total_num = 500000
num_O = 250000
num_non_O = total_num - num_O

percents = {
    'DATE':0.15,
    'CARDINAL':0.1,
    'DECIMAL':0.1,
    'MEASURE':0.1, 
    'MONEY':0.1, 
    'ORDINAL':0.1, 
    'TIME':0.05, 
    'DIGIT':0.1, 
    'FRACTION':0.05, 
    'TELEPHONE':0.1, 
    'ADDRESS':0.05
}

final_data = []

data_O = label_group["O"]
random.shuffle(data_O)
final_data.extend(data_O[:num_O])

label_count = { label: 0 for label in label_list }

for k,v in percents.items():
    label_data = label_group[k]
    random.shuffle(label_data)
    num_exmaple = int(v * num_non_O)
    # print(f"num_exmaple: {num_exmaple}")
    index = 0
    while label_count[k] < num_exmaple:
        final_data.append(label_data[index])
        index += 1
        for l in label_data[index]["labels"]:
            label_count[l] += 1
    # final_data.extend(label_data[:num_exmaple])
    
count_all = { label: 0 for label in label_list }
count_all["O"] = 0
for d in final_data:
    labels = d["labels"]
    if labels != []:
        for l in labels:
            count_all[l] += 1
    else:
        count_all["O"] += 1

for k, v in count_all.items():
   print(f"{k}: {v}") 

random.shuffle(final_data)
with open("./data/data.json","w") as f:
    json.dump(final_data,f,indent=2,ensure_ascii=False)
    
train_data = final_data[:int(total_num*0.9)]
dev_data = final_data[int(total_num*0.9):int(total_num*0.95)]
test_data = final_data[int(total_num*0.95):]

with open("./data/train.json","w") as f:
    json.dump(train_data,f,indent=2,ensure_ascii=False)
    
with open("./data/dev.json","w") as f:
    json.dump(dev_data,f,indent=2,ensure_ascii=False)
    
with open("./data/test.json","w") as f:
    json.dump(test_data,f,indent=2,ensure_ascii=False)