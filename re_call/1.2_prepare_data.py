# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os
f=open("df_train.pkl",mode="rb")
df_train=pickle.load(f)


uid_id_dict={}
id_uid_dict={}
df=df_train
item_id_dict={}
id_item_dict={}
cate_id_dict={}
id_cate_dict={}
action_id_dict={}
id_action_dict={}
for i,j in  enumerate(df["uid"].drop_duplicates().values):
    uid_id_dict[j]=i
    id_uid_dict[i]=j
for i,j in  enumerate(df["itemid"].drop_duplicates().values):
    item_id_dict[j]=i
    id_item_dict[i]=j


for i,j in  enumerate(df["cateid"].drop_duplicates().values):
    cate_id_dict[j]=i
    id_cate_dict[i]=j



df_train["uid"]=df_train["uid"].map(uid_id_dict)
df_train["itemid"]=df_train["itemid"].map(item_id_dict)
df_train["cateid"]=df_train["cateid"].map(cate_id_dict)
df_train_date=df_train[["uid","itemid","cateid","action","time"]]

with open("train.data",mode="wb") as f:
    pickle.dump((uid_id_dict,id_uid_dict,item_id_dict,id_item_dict,cate_id_dict,id_cate_dict,df_train_date),f )




