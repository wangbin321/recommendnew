# -*- coding:utf-8 -*-
import pandas as pd
import  pickle
import numpy as np
import tensorboard
df = pd.read_csv("/Users/wangbin/Documents/DeepFM/data_1.csv")
df.columns = ["uid", "itemid", "cateid", "action", "time"]

for i in df.columns:
    df[i] = df[i].map(str)
import time
def timeutil(t):
    t = int(t)
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
df["time"] = df["time"].map(timeutil)
df["rank"]=df.groupby(["uid"])["time"].apply(lambda x:x.rank(ascending=False))
df_target=df[df["rank"]==1]
with open("df_target.pkl",mode="wr") as f:
   pickle.dump(df_target,f,protocol=2)

df_train=df[df["rank"]!=1]
with open("df_train.pkl",mode="wr") as f:
   pickle.dump(df_train,f,protocol=2)