import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential

def get_model():
    inputs = Input(shape=(3,))
    x = Dense(3, activation='relu')(inputs)
    x1 = Dense(1, activation='softmax')(x)
    x2 = Dense(1, activation='softmax')(x1)

    return Model(inputs, outputs= [x1,x2])

def loss(y_true, y_pred):
    o1 = y_pred[0]
    o2 = y_pred[1]
    y = tf.cast(y_true, dtype='float')
    with tf.compat.v1.Session() as sess:
        print((o1+o2-y).eval())
    return o1 + o2 - y

x = np.array([[1,2,3],[1,2,1]])
y = np.array([1,0])

model = get_model()
model.compile(optimizer='adam', loss=loss, metrics=[])
model.fit(x, y, epochs=1)
