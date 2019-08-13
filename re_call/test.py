import pandas as pd
import numpy as np

# a=tf.constant([1,2,3,4],dtype=tf.float32)
# b=tf.constant([2,2,3,4],dtype=tf.float32)
# b_1=tf.transpose(b)
# c=tf.multiply(a,b)
# d=tf.matmul(a,b_1,transpose_b=True)
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(c))
#
#     print(sess.run(d)


df=pd.DataFrame({"a":range(0,10,1),"b":np.random.normal(size=10)})

for i in range(0,len(df),3):
    print df[i:i+3]
df.sample()