# -*- coding:utf-8 -*-
import tensorflow as tf
import pickle
import pandas as pd
import  numpy as np
import os
m=pickle.load(open("train.data",mode="rb"))
[uid_id_dict,id_uid_dict,item_id_dict,id_item_dict,cate_id_dict,id_cate_dict,df_train_date]=m
user_size=len(uid_id_dict)+1
item_size=len(item_id_dict)+1
class NCF(object):
    def __init__(self,user_size,item_size):
        self.user_size=user_size
        self.item_size=item_size
        self.embed_size=128
        self.activation_func=tf.nn.leaky_relu
        self.initializer=tf.truncated_normal_initializer(stddev=0.01)
        self.regularizer= tf.contrib.layers.l2_regularizer(0.002)


    def run(self):
          self.user=[1,1,2,3,4]
          self.item=[2,2,3,1,2]
          self.label=[1.0,0,1.0,0,0]


    def create_model(self):
        with tf.name_scope("onet_hot"):
            self.user_onehot = tf.one_hot(self.user, self.user_size, name='user_onehot')
            self.item_onehot = tf.one_hot(self.item, self.item_size, name='item_onehot')

        with tf.name_scope('embed'):
            self.user_embed_GMF = tf.layers.dense(inputs = self.user_onehot,
                                                  units = self.embed_size,
                                                  activation = self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_GMF')

            self.item_embed_GMF = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_GMF')

            self.user_embed_MLP = tf.layers.dense(inputs=self.user_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_MLP')
            self.item_embed_MLP = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_MLP')
        with tf.name_scope("GMF"):
            self.GMF=tf.multiply(self.user_embed_GMF,self.item_embed_GMF,name="GMF")
        with tf.name_scope("MLP"):
            self.interaction = tf.concat([self.user_embed_MLP, self.item_embed_MLP],
                                         axis=-1, name='interaction')
            self.MLP=tf.layers.dense(inputs=self.interaction,units=self.embed_size*2,kernel_initializer=self.initializer,
                                     activation=self.activation_func,
                                     kernel_regularizer=self.regularizer
                                     )
            self.MLP=tf.nn.dropout(self.MLP,keep_prob=0.5)

            self.MLP = tf.layers.dense(inputs=self.interaction, units=self.embed_size ,
                                       kernel_initializer=self.initializer,
                                       activation=self.activation_func,
                                       kernel_regularizer=self.regularizer
                                       )
            self.MLP = tf.nn.dropout(self.MLP, keep_prob=0.5)

            self.MLP = tf.layers.dense(inputs=self.interaction, units=self.embed_size//2,
                                       kernel_initializer=self.initializer,
                                       activation=self.activation_func,
                                       kernel_regularizer=self.regularizer
                                       )
            self.MLP = tf.nn.dropout(self.MLP, keep_prob=0.5)
        with tf.name_scope("concat"):
            self.concat=tf.concat([self.GMF,self.MLP],axis=-1,name="concat")
            self.logits = tf.layers.dense(inputs=self.concat,
                                          units=1,
                                          activation=None,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          name='predict')

            self.logits_dense = tf.reshape(self.logits, [-1])

        with tf.name_scope("loss"):
            self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label,logits=self.logits_dense,name="loess")
            self.mean_loss=tf.reduce_mean(self.loss)

        with tf.name_scope("optimzation"):
            self.optimzer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.mean_loss)

        self.writer = tf.summary.FileWriter('./graphs/NCF', tf.get_default_graph())
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self,sess,gloal_steps):
         # feed_dict={self.user:uid,self.item:item,self.label:label}
         mean_loss, optim, summaries = sess.run(
             [self.mean_loss, self.optimzer, self.summary_op])
         self.writer.add_summary(summaries, global_step=gloal_steps)

         return  mean_loss


if __name__=="__main__":
    m=pickle.load(open("train.data",mode="rb"))
    [uid_id_dict,id_uid_dict,item_id_dict,id_item_dict,cate_id_dict,id_cate_dict,df_train_date]=m
    df_train_date["label"]=df_train_date["action"].map(lambda x:x!='pv').astype(int)
    uid_size=len(uid_id_dict)+1
    item_size=len(item_id_dict)+1
    df_train_date1=df_train_date[["uid","itemid","label"]]
    for i in df_train_date1.columns:
        df_train_date1[i]=df_train_date1[i].astype(int)
    df_train_date1=df_train_date1.sample(len(df_train_date1),replace=False)
    del df_train_date
    total=len(df_train_date1)
    batch_size=2048
    model_path="./model"





    model=NCF(uid_size,item_size)
    model.run()
    model.create_model()
    count=0
    with tf.Session()  as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(0,total,batch_size):
           data=df_train_date1[i:i+batch_size]
           user=data[["uid"]].values
           item=data[["itemid"]].values
           label=data["label"].values


           model.run()
           model.user=user
           model.item=item
           model.label=label

           mean_loss=model.train(sess,count)
           count=count+1
           print "index %d mean loss %f " % i,mean_loss
        checkpoint_path = os.path.join(model_path, "NCF.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=count)






