# -*- coding:utf-8 -*-
import pickle
import pandas as pd
import numpy as np

[uid_id_dict, id_uid_dict, item_id_dict, id_item_dict, cate_id_dict, id_cate_dict, df_train_date] = pickle.load(
    open("train.data", mode="rb"))
uidkeys=list(uid_id_dict.keys())
total=len(uidkeys)

def f(df):
    def gener(x,y,target):
        result=[]
        if  len(x)==0 or len(y)==0:
            return pd.DataFrame()
        for i in x:
            for j in y:
                tmp=[]
                tmp.append(i)
                tmp.append(j)
                tmp.append(target)
                result.append(tmp)
        df_tmp=pd.DataFrame(result)
        df_tmp.columns=["item1","item2","target"]
        return df_tmp
    df.sort_values(by=["target"],ascending=False,inplace=True)
    target_list=sorted(list(set((df["target"].values))))
    df_list=[]
    for  i in range(0,len(target_list)-2):
        a=target_list[0]
        b=target_list[1]
        a_item=df[df["target"]==a]["item"].values
        b_item=df[df["target"]==b]["item"].values
        tmp1=gener(a_item,b_item,1 )
        if not tmp1.empty:
          df_list.append(tmp1)
    tmp2=gener(list(set((np.random.choice(df[df["target"]==0]["item"].values,size=100,replace=True)))),df[df["target"]!=0]["item"].values,0)
    if not tmp2.empty:
        df_list.append(tmp2)
    if len(df_list)>0:
        return pd.concat(df_list,axis=0)
    return pd.DataFrame()
target_dict={'pv':0,'cart':1,'fav':2,'buy':3}
print(df_train_date["action"].value_counts())
df_train_date["target"]=df_train_date["action"].map(target_dict)
df_train_date=df_train_date[["uid","itemid","target"]]
df_train_date.columns=["uid","item","target"]

size=20
num=int(total/size)

for i in range(0,size+1):
    tmp_keys=uidkeys[i*num:(i*num+num)]
    if tmp_keys and len(tmp_keys)>0:
        df_train_date1=df_train_date[df_train_date["uid"].isin(tmp_keys)]
        if not  df_train_date1.empty and len(df_train_date1)>0:
            result =df_train_date1.groupby(["uid"]).apply(lambda  x:f(x)).reset_index()
            if not result.empty:
                result = result[["uid", "item1", "item2", "target"]]
                result.to_csv("BPR/BPR"+str(i)+".csv")


