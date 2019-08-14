# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle




class BPR(object):
    def __init__(self,uid_size,item_size,hidden_dim,learn_rate):
        self.uid_size=uid_size
        self.item_size=item_size
        self.hidden_dim=hidden_dim
        self.learn_rate=learn_rate
        self.get_placeholder()
        self.get_embedding()
        self.create_model()
        self.get_optimizer()
    def  get_placeholder(self):
         self.uid=tf.placeholder(tf.int32,name="uid")
         self.item1=tf.placeholder(tf.int32,name="item1")
         self.item2 = tf.placeholder(tf.int32, name="item2")
         self.target=tf.placeholder(tf.int32,name="target")
    def get_embedding(self):
        self.uid_embedding=tf.Variable(tf.truncated_normal(shape=(self.uid_size,self.hidden_dim),mean=0,stddev=0.01),name="uid_embedding")
        self.item_embedding=tf.Variable(tf.truncated_normal(shape=(self.item_size,self.hidden_dim),mean=0,stddev=0.01),name="item_embedding")


    def create_model(self):
        self.uid_feature=tf.nn.embedding_lookup(self.uid_embedding,self.uid)
        self.item1_feature=tf.nn.embedding_lookup(self.item_embedding,self.item1)
        self.item2_feature=tf.nn.embedding_lookup(self.item_embedding,self.item2)

        self.loss=tf.reduce_mean(tf.matmul(self.uid_feature,tf.subtract(self.item1_feature,self.item2_feature),transpose_b=True),1)
        self.loss=tf.nn.softmax(self.loss)

        self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target,logits=self.loss)
        self.saver = tf.train.Saver(tf.global_variables())


    def get_optimizer(self):
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

if __name__=="__main__":
    dir=os.listdir("BPR")
    uid_szie=19544
    item_size=625174

    hidden_dim=256
    batch_size=512

    model=BPR(uid_szie,item_size,hidden_dim,0.05)
    model_path="BPRModel/model"
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        count=1
        count_size=0
        for x in range(1000):
            for file in dir:
               print("read file "+file)
               for i in   pd.read_csv("BPR/"+file,chunksize=batch_size):
                   uid=i["uid"].values
                   item1=i["item1"].values
                   item2=i["item2"].values
                   target=i["target"].values
                   feed_dict={model.uid:uid,model.item1:item1,model.item2:item2,model.target:target}

                   loss,_=sess.run([model.loss,model.optimizer],feed_dict=feed_dict)
                   count_size=0
                   if count_size%100==0:
                       print(loss)
            checkpoint_path = os.path.join(model_path, "BPR.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=count)
            count=count+1













