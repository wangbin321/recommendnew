# -*- coding:utf-8 -*-
from surprise import SVDpp,SVD,Reader,Dataset
import pickle

import pandas as pd
import numpy as np

"""
import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                'userID': [9, 32, 2, 45, 'user_foo'],
                'rating': [3, 2, 4, 3, 1]}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(NormalPredictor(), data, cv=2)
"""
m=pickle.load(open("train.data",mode="rb"))
[uid_id_dict,id_uid_dict,item_id_dict,id_item_dict,cate_id_dict,id_cate_dict,df_train_date]=m
print(df_train_date["action"].value_counts())
model=SVDpp()
action_dict={}
action_dict['pv']=0
action_dict['cart']=1
action_dict["fav"]=2
action_dict["buy"]=3
df_train_date["action"]=df_train_date["action"].map(action_dict)
reader=Reader(rating_scale=(1,4))
data=Dataset.load_from_df(df_train_date[["uid","itemid","action"]],reader=reader)
data=data.build_full_trainset()
model.fit(data)
with  open("svdpp.model",mode="wb") as f:
    pickle.dump(model,f)
f=open("svdpp.model",mode="rb")
model=pickle.load(f)
u_mat=model.pu
i_mat=model.qi

f1=open("mat.dat",mode="wb")
pickle.dump((u_mat,i_mat),f1)