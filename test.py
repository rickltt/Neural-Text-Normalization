from transformers import pipeline, AutoTokenizer
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# tokenizer = AutoTokenizer.from_pretrained("tagger_output")
# pipe = pipeline("token-classification","tagger_output",device="cpu")
# print(pipe("The Man has 2.8 million dollars, his telephone number is 0-393-04994-9."))

# pipe_summary = pipeline("summarization","decoder_output",device="cpu")
# print(pipe_summary("normalize DATE: 29 April 2013", min_length=5, max_length=32, length_penalty=2.0, num_beams=4, early_stopping=True))

# tokens = tokenizer.tokenize("The Man has 2.8 million dollars, his telephone number is 0-393-04994-9.")
# print(tokens)

def read_df(file):

    df = pd.read_csv(file,index_col=False)
    df = df[~df["Output Token"].isin(["<self>","sil"])]
    # print(df["Semiotic Class"].unique())
    df = df[df["Semiotic Class"].isin(['DATE', 'CARDINAL', 'DECIMAL','MEASURE', 'MONEY', 'ORDINAL', 'TIME', 'DIGIT', 'FRACTION', 'TELEPHONE', 'ADDRESS'])]
    df = df.drop_duplicates()
    # df.to_csv("./output_1.csv",index=False)
    # print(df.head())
    # print(df.info())
    return df

df = pd.DataFrame()
files = ["output_1.csv", "output_6.csv", "output_11.csv", "output_16.csv", "output_91.csv", "output_96.csv"]
for file in files:
    df = read_df(os.path.join("dataset",file))
    df = pd.concat([df,df])

print(df["Semiotic Class"].value_counts())
# train_df.to_csv("./dataset/train_decoder.csv",index=False)
train, dev_test = train_test_split(df, test_size=0.1,random_state=42, stratify=df["Semiotic Class"])
dev, test = train_test_split(dev_test, test_size=0.5,random_state=42, stratify=dev_test["Semiotic Class"])

print(train["Semiotic Class"].value_counts())
print(dev["Semiotic Class"].value_counts())
print(test["Semiotic Class"].value_counts())

train.to_csv("./dataset/train_decoder.csv",index=False)
dev.to_csv("./dataset/dev_decoder.csv",index=False)
test.to_csv("./dataset/test_decoder.csv",index=False)
# dev_df = read_df(os.path.join("dataset","output_91.csv"))
# dev_df.to_csv("./dataset/dev_decoder.csv",index=False)

# test_df = read_df(os.path.join("dataset","output_96.csv"))
# test_df.to_csv("./dataset/test_decoder.csv",index=False)
