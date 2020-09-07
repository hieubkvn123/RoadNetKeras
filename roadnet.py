import os
import numpy as np
import tensorflow as tf

from models import SideNet
from models import RoadSurfaceNet
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class RoadNet(object):
    def __init__(self, input_shape=(128,128,1)):
        self.input_shape=input_shape
        self.centerline_net = SideNet(name='centerline', input_shape=input_shape)
        self.edge_net = SideNet(name='edge', input_shape=input_shape)
        self.surface_net = RoadSurfaceNet(input_shape=input_shape)

    def loss(self, y_true, y_pred):
        mask_surface, mask_edge, mask_line = y_true
        s1_surface, s2_surface, s3_surface, s4_surface, fused_surface, s1_edge, s2_edge, s3_edge, s4_edge, fused_edge, s1_line, s2_line, s3_line, s4_line, fused_line = y_pred

        

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        s1_surface, s2_surface, s3_surface, s4_surface, s5_surface, fused_surface = self.surface_net.get_model()(inputs)
        s1_edge, s2_edge, s3_edge, s4_edge, fused_edge = self.edge_net.get_model()([inputs, fused_surface])
        s1_line, s2_line, s3_line, s4_line, fused_line = self.centerline_net.get_model()([inputs, fused_surface])

        model = Model(inputs=inputs, outputs=[s1_surface, s2_surface, s3_surface, s4_surface, s5_surface, fused_surface, 
            s1_edge, s2_edge, s3_edge, s4_edge, fused_edge, s1_line, s2_line, s3_line, s4_line, fused_line])

        return model
