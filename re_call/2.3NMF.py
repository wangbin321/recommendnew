# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import json
from scipy.sparse import coo_matrix
import pickle


class NFM(object):
    def __init__(self,file_path):
             m=pickle.load(open(file_path,mode="rb"))
             [uid_id_dict,id_uid_dict,item_id_dict,id_item_dict,cate_id_dict,id_cate_dict,df_train_date]=m
             self.uid_id_dict=uid_id_dict
             self.item_id_dict=item_id_dict
             df_train_date["target"]=df_train_date["action"].map(lambda x:x!='pv').astype(float)
             self.shape=(512,512)

             self.data=df_train_date[["uid","itemid","target"]].values
             del df_train_date
             self.total_size=len(self.data)
             np.random.shuffle(self.data)
             self.userLayer=[256,128]
             self.itemLayer=[256,128]
             self.batch_size=1024
             self.add_placeholders()
             self.get_embedding()
             self.add_model()

    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.int32)
    def get_embedding(self):
        self.uid_embedding=tf.get_variable(name="uid_embedding",shape=[len(self.uid_id_dict),self.shape[0]],
                                           initializer=tf.random_normal_initializer(mean=0,stddev=0.01))
        self.item_embedding=tf.get_variable(name="item_embedding",shape=[len(self.item_id_dict),self.shape[0]],\
                                           initializer=tf.random_normal_initializer(mean=0,stddev=0.01))

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.uid_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

            norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
            norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
            self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
            self.y_ = tf.maximum(1e-6, self.y_)
        with tf.name_scope("loss"):
            self.logit=tf.nn.softmax(self.y_)
            self.logit=tf.reshape(self.logit,shape=[-1])
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.rate,logits=self.logit))
        with tf.name_scope("optimizer"):
            self.optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)


    def run(self,sess):
        model_path="./model/DMF"
        for j in range(0,3):
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Creating model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
            for i in range(0,self.total_size,self.batch_size):
                data=self.data[i:i+self.batch_size]
                uid=data[:,0]
                item=data[:,1]
                label=data[:,2]
                feed_dict={self.user:uid,self.item:item,self.rate:label}

                loss,_= sess.run([self.loss,self.optimizer],feed_dict=feed_dict)

                print "index "+str(i*(j+1))+" mean loss  " +str(loss)
            checkpoint_path = os.path.join(model_path, "NCF.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=j)

model=NFM("train.data")
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    model.run(sess)




