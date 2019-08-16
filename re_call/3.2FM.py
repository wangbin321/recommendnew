# -*- coding:utf-8 -*-
import os
import pandas as pd
import  tensorflow as tf
import numpy as np
import gc
import logging
logging.basicConfig(level=logging.DEBUG,
                    filename='FM.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

                    )

class  FM(object):
     def __init__(self,feature_size,lr,dim):
         self.feature_size=feature_size
         self.lr=lr
         self.dim=dim
         self.create_mode()
     def create_mode(self):
         self.x=tf.placeholder(tf.float32,shape=[None,self.feature_size],name="x")
         self.y=tf.placeholder(tf.float32,name="y")

         self.embedding=tf.Variable(tf.truncated_normal(shape=(self.feature_size,self.dim),mean=0.0,stddev=0.001),name="eembedding")
         self.w=tf.Variable(tf.truncated_normal(shape=[self.feature_size,1],mean=0.0,stddev=0.001),name="w")
         self.w0=tf.Variable(1.0,name='w0')

         ## frist=

         self.liner=tf.matmul(self.x,self.w)+self.w0

         # shape of [None, 1]
         self.interaction_terms = tf.multiply(0.5,
                                         tf.reduce_mean(
                                             tf.subtract(
                                                 tf.pow(tf.matmul(self.x, self.embedding), 2),
                                                 tf.matmul(self.x, tf.pow(self.embedding, 2))),
                                             1, keep_dims=True))

         self.out=self.liner+self.interaction_terms
         self.out=tf.nn.sigmoid(self.out)
         self.loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.out)
         self.loss=tf.reduce_mean(self.loss)
         self.optimizer=tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
         self.saver=tf.train.Saver(max_to_keep=3)
def generfeature( df,uidfeather_dict,itemfeather_dict,uid_szie = 19544,tem_size = 50000,):
    total_len=len(df)


    df["action"] = df["action"].map(lambda x:x!='pv').astype(int)
    df["itemid"]=df["itemid"].map(lambda x:x%tem_size)
    feature=np.zeros(shape=[total_len,uid_szie+tem_size])
    uid_vec=[]
    item_vec=[]
    for (index,value) in enumerate(df["uid"].values):
        feature[index][value]=1
        while value not in uidfeather_dict:
            value=value-1
        uid_vec.append(uidfeather_dict[value])

    for (index,value) in enumerate(df["itemid"].values):
        feature[index][value+uid_szie]=1
        while   value  not in itemfeather_dict:
            value=value-1
        item_vec.append(itemfeather_dict[value])
    x=np.concatenate([feature,uid_vec,item_vec],axis=1)
    y=df["action"].values
    return x,y
if __name__=="__main__":
    model_dir="./FM"
    feature_size=69549
    echo=100
    batch_size=512
    model=FM(feature_size,lr=0.05,dim=8)
    uidfeather = np.load("uid",
                         allow_pickle=True)
    uidfeather = pd.DataFrame(uidfeather)
    uidfeather.columns = ["uid", "uid_1", "uid_2", "uid_3"]
    uidfeather=uidfeather.values
    uidfeather_dict={}
    for i in uidfeather:
        uidfeather_dict[i[0]]=i[1:]


    itemfeather = np.load("item",
                          allow_pickle=True)
    itemfeather = pd.DataFrame(itemfeather)
    itemfeather.columns = ["itemid", "item_1", "item_2"]
    itemfeather=itemfeather.values
    itemfeather_dict={}
    for i in itemfeather:
        itemfeather_dict[i[0]]=i[1:]
    del  itemfeather ,uidfeather
    gc.collect()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        count=0
        for index in range(0,2) :
            for df in pd.read_csv("df_train_date.csv",chunksize=1024):
                    x,y =generfeature(df,uidfeather_dict,itemfeather_dict)
                    feed_dict1={model.x:x,model.y:y}
                    count=count+len(x)
                    loss,_=sess.run([model.loss,model.optimizer],feed_dict=feed_dict1)
                    logging.info("iter:%d,count:%d,  loss: %6.5f" % (index,count, loss))
                    if count!=0 and count%100000==0:
                        model.saver.save(sess=sess, save_path=model_dir, global_step=count)
            model.saver.save(sess=sess,save_path=model_dir,global_step=count)


