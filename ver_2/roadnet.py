import os
import sys
import numpy as np
import tensorflow as tf

from models import SideNet
from models import RoadSurfaceNet
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class RoadNet(object):
    def __init__(self, input_shape=(128,128,3)):
        self.input_shape=input_shape
        self.centerline_net = SideNet(name='centerline')
        self.edge_net = SideNet(name='edge')
        self.surface_net = RoadSurfaceNet()
        self.beta = 0.99 ### For balanced cross-entropy ###
        self.lambda_ = 2e-4 ### For generalization ###
        
        ### For weights of the loss components ###
        self.alpha = 1.0 # for balanced cross-entropy
        self.gamma = 1.0 # for regularization 
        self.eta   = 1.0 # for generalization

    def cross_entropy_balanced(self, y_true, y_pred):
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        # y_pred = tf.math.log(y_pred/(1-y_pred))
        # y_pred = (tf.nn.sigmoid(y_pred) - 0.5)/0.5
        y_pred = -tf.math.log((1/tf.nn.sigmoid(y_pred)) - 1)

        y_true = tf.cast(y_true, tf.float32)

        count_neg = tf.reduce_sum(1. - y_true)
        count_pos = tf.reduce_sum(y_true)

        ### Calc beta ###
        beta = count_neg/(count_neg+count_pos)

        ### Calc pos_weight ###
        pos_weight = beta / (1-beta)

        cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
        # cost = tf.reduce_mean(cost * (1 - beta))

        return cost 

    def weighted_binary_crossentropy(self): 
        def loss(y_true, y_pred):
            # y_pred = tf.math.sigmoid(y_pred)
            _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
            y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
            # y_pred = -tf.math.log((1/y_pred) - 1)

            y_true = tf.cast(y_true, tf.float32)
            # transform predicted map to a probability map
            # y_pred = tf.nn.sigmoid(y_pred)
            count_neg = tf.math.reduce_sum(1 - y_true)
            count_pos = tf.math.reduce_sum(y_true)
            beta = count_neg / (count_neg + count_pos)
           
            w1 = beta 
            w2 = 1 - beta
            
            # ones = tf.ones_like(y_true)
            # msk = tf.equal(y_true, ones)
            ### Calculate weighted binary cross-entropy loss with beta=0.1 to signify the importance of loss where y==1 ###
            # res, _ = tf.map_fn(lambda x: (tf.multiply(-tf.math.log(x[0]), w1) if x[1] is True else tf.multiply(-tf.math.log(1 - x[0]), w2), x[1]),
            #                   (y_pred, msk), dtype=(tf.float32, tf.bool)) 
            # res = - beta * tf.multiply(tf.math.log(y_pred), y_true) - (1-beta) * tf.multiply(tf.math.log(1-y_pred), 1-y_true) 
            # res = tf.math.reduce_mean(res)
            b_ce = K.binary_crossentropy(y_true, y_pred)
            weight_vector = y_true * beta + (1. - y_true) * (1-beta)
            weighted_b_ce = weight_vector * b_ce

            loss_pos = weighted_b_ce * y_true
            loss_neg = weighted_b_ce * (1 - y_true)

            # res = K.mean(weighted_b_ce)
            res = K.mean(loss_pos) + K.mean(loss_neg)

            # res = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)
            ### L2 normalization ###
            ### l2 norm = 1/(2|X|) * ||Y- P||2
            # l2_norm = tf.reduce_mean((tf.sigmoid(y_pred) - y_true)**2) * 0.5
    
            return res # + l2_norm

        return loss
    
    def weighted_binary_crossentropy_with_l2(self): 
        def loss(y_true, y_pred):
            # res = self.cross_entropy_balanced(y_true, y_pred)
            res = self.weighted_binary_crossentropy()(y_true, y_pred)

            ### L2 normalization ###
            ### l2 norm = 1/(2|X|) * ||Y- P||2
            
            _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
            y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
            # y_pred = tf.math.log(y_pred/(1-y_pred))

            y_true = tf.cast(y_true, dtype=tf.float32)

            l2_norm = tf.reduce_mean((y_pred - y_true) ** 2) * 0.5

            return res + l2_norm

        return loss

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        s1_surface, s2_surface, s3_surface, s4_surface, s5_surface, fused_surface = self.surface_net.get_model()(inputs)
        s1_edge, s2_edge, s3_edge, s4_edge, fused_edge = self.edge_net.get_model()([inputs, fused_surface])
        s1_line, s2_line, s3_line, s4_line, fused_line = self.centerline_net.get_model()([inputs, fused_surface])

        ### Basically naming layer for loss monitoring ###
        fused_surface = Lambda(lambda x: x, name='surface_final_output')(fused_surface)
        fused_edge    = Lambda(lambda x: x, name='edge_final_output')(fused_edge)
        fused_line    = Lambda(lambda x: x, name='line_final_output')(fused_line)
        
        s1_surface = Lambda(lambda x : x, name='surface_side_output_1')(s1_surface)
        s2_surface = Lambda(lambda x : x, name='surface_side_output_2')(s2_surface)
        s3_surface = Lambda(lambda x : x, name='surface_side_output_3')(s3_surface)
        s4_surface = Lambda(lambda x : x, name='surface_side_output_4')(s4_surface)
        s5_surface = Lambda(lambda x : x, name='surface_side_output_5')(s5_surface)

        s1_edge = Lambda(lambda x : x, name='edge_side_output_1')(s1_edge)        
        s2_edge = Lambda(lambda x : x, name='edge_side_output_2')(s2_edge)        
        s3_edge = Lambda(lambda x : x, name='edge_side_output_3')(s3_edge)        
        s4_edge = Lambda(lambda x : x, name='edge_side_output_4')(s4_edge) 

        s1_line = Lambda(lambda x : x, name='line_side_output_1')(s1_line)
        s2_line = Lambda(lambda x : x, name='line_side_output_2')(s2_line)
        s3_line = Lambda(lambda x : x, name='line_side_output_3')(s3_line)
        s4_line = Lambda(lambda x : x, name='line_side_output_4')(s4_line)
        
        model = Model(inputs=inputs, outputs=[fused_surface, fused_edge, fused_line, s1_surface, s2_surface, s3_surface, s4_surface,s5_surface, \
                s1_edge, s2_edge, s3_edge, s4_edge, \
                s1_line, s2_line, s3_line, s4_line])

        return model


