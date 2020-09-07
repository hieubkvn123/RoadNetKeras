import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class RoadSurfaceNet(object):
    def __init__(self, input_shape=(128,128,3)):
        self.input_shape=input_shape
        
    def selu(self, x):
        alpha = 1.67326324
        scale = 1.05070098

        mask_1 = tf.cast((x > 0), dtype=tf.float32)
        mask_2 = tf.cast((x < 0), dtype=tf.float32)

        selu_ = (x * scale * mask_1) + ((scale * alpha * (tf.math.exp(x) - 1)) * mask_2)

        return selu_

    def get_model(self):
        ### Stage 1 ###
        inputs = Input(shape=self.input_shape)
        conv_1 = Conv2D(64, kernel_size=(3,3), activation=self.selu, padding='same')(inputs)
        conv_1 = Conv2D(64, kernel_size=(3,3), activation=self.selu, padding='same')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        ### First output ###
        ### Size = W * H * 1 ###
        side_1 = Conv2D(1, kernel_size=(1,1), activation='relu', padding='same', name='output_1')(conv_1)

        ### Stage 2 ###
        conv_2 = Conv2D(128, kernel_size=(3,3), activation=self.selu, padding='same')(pool_1)
        conv_2 = Conv2D(128, kernel_size=(3,3), activation=self.selu, padding='same')(conv_2)
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        ### Second output ###
        ### Size = W/2 * H/2 * 1 ###
        side_2 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_2)
        side_2 = Conv2DTranspose(1, kernel_size=(2,2), strides=(2,2), padding='same', name='output_2')(side_2)

        ### Stage 3 ###
        conv_3 = Conv2D(256, kernel_size=(3,3), activation=self.selu, padding='same')(pool_2)
        conv_3 = Conv2D(256, kernel_size=(3,3), activation=self.selu, padding='same')(conv_3)
        conv_3 = Conv2D(256, kernel_size=(3,3), activation=self.selu, padding='same')(conv_3)
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        ### Third output ###
        ### Size = W/4 * H/4 * 1 ###
        side_3 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_3)
        side_3 = Conv2DTranspose(1, kernel_size=(4,4), strides=(4,4), padding='same', name='output_3')(side_3)

        ### Stage 4 ###
        conv_4 = Conv2D(512, kernel_size=(3,3), activation=self.selu, padding='same')(pool_3)
        conv_4 = Conv2D(512, kernel_size=(3,3), activation=self.selu, padding='same')(conv_4)
        conv_4 = Conv2D(512, kernel_size=(3,3), activation=self.selu, padding='same')(conv_4)
        pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

        ### Fourth output ###
        ### Size = W/8 * H/8 * 1 ###
        side_4 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_4)
        side_4 = Conv2DTranspose(1, kernel_size=(8,8), strides=(8,8), padding='same', name='output_4')(side_4)

        ### Stage 5 ###
        conv_5 = Conv2D(512, kernel_size=(3,3), activation=self.selu, padding='same')(pool_4) 
        conv_5 = Conv2D(512, kernel_size=(3,3), activation=self.selu, padding='same')(conv_5)
        conv_5 = Conv2D(512, kernel_size=(3,3), activation=self.selu, padding='same')(conv_5)

        ### Fifth output ###
        ### Size = W/16 * H/16 * 1 ###
        side_5 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_5)
        side_5 = Conv2DTranspose(1, kernel_size=(16,16), strides=(16,16), padding='same', name='output_5')(side_5)

        ### concatenated output ###
        print(side_1.shape, side_2.shape, side_3.shape, side_4.shape, side_5.shape)
        concat = tf.keras.layers.concatenate((side_1, side_2, side_3, side_4, side_5), axis=3)
        concat = Conv2D(1, kernel_size=(1,1), activation='relu', name='final_concat')(concat)

        model = Model(inputs=inputs, outputs=[side_1,side_2,side_3,side_4,side_5,concat])

        return model
