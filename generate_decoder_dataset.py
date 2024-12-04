import pandas as pd
import os
from sklearn.model_selection import train_test_split

def read_df(file):

    df = pd.read_csv(file,index_col=False)
    df = df[df["Input Token"].notna()]
    df = df[~df["Output Token"].isin(["<self>","sil"])  & df["Output Token"].notna() ]
    df = df.drop_duplicates()
    return df

df = pd.DataFrame()
files = ["output_1.csv", "output_6.csv", "output_11.csv", "output_16.csv", "output_91.csv", "output_96.csv"]
for file in files:
    df = read_df(os.path.join("dataset",file))
    df = pd.concat([df,df])
print("-"*25 + "Total" + "-"*25)
print(df["Semiotic Class"].value_counts())

train, dev_test = train_test_split(df, test_size=0.1,random_state=42, stratify=df["Semiotic Class"])
dev, test = train_test_split(dev_test, test_size=0.5,random_state=42, stratify=dev_test["Semiotic Class"])

print("-"*25 + "Train" + "-"*25)
print(train["Semiotic Class"].value_counts())

print("-"*25 + "Dev" + "-"*25)
print(dev["Semiotic Class"].value_counts())

print("-"*25 + "Test" + "-"*25)
print(test["Semiotic Class"].value_counts())

train.to_csv("./dataset/train_decoder.csv",index=False)
dev.to_csv("./dataset/dev_decoder.csv",index=False)
test.to_csv("./dataset/test_decoder.csv",index=False)
