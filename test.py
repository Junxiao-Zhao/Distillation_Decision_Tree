import lightgbm as lgb
import tensorflow as tf
""" import tensorflow.keras as keras
from tensorflow.keras.datasets import boston_housing

def a(fst, snd, thd):
    print(fst, snd, thd)

def b(ha, **kwargs):
    #print(ha)
    a(fst=ha, **kwargs)

#b(ha="start", snd="2", thd="3")

layer = keras.layers.LayerNormalization(axis=1)

data = tf.constant([[1,2,3,4,5,6],
        [0.1, 0.7,5,9,0.01,100]])

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.mean(axis=0)) """

train_data = lgb.Dataset('train.svm.bin')