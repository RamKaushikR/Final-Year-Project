import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend as K
from model import Conv1D, Conv2D, LSTM


class Siamese(object):
    def __init__(self, input_shape, lr=0.001, core_path=None, model_path=None, name='conv_1d', margin=1):
        self.w = input_shape[0]
        self.h = input_shape[1]
        
        self.core = self.createCore(name)
            
        self.model = self.createModel(name, input_shape, lr, margin, core_path=core_path)
        
        if model_path:
            self.model.load_weights(model_path)
        if core_path:
            self.model.get_layer('sequential').load_weights(core_path, by_name=True)
            for layer in self.model.get_layer('sequential').layers[:-3]:
                layer.trainable = True
            # self.model.get_layers('sequential').trainable = False
            # self.core.load_weights(core_path)
            # self.core.trainable = False
    
    def createCore(self, name='conv_1d'):
        if name == 'conv_1d':
            core = Conv1D()
            
        elif name == 'conv_2d':
            core = Conv2D()
            
        else:
            core = LSTM()
            
        core.add(layers.Dense(256, activation='relu', activity_regularizer=l2(0.001), name='fc_dense_1'))
        core.add(layers.Dense(192, activation='sigmoid', name='fc_dense_2'))
        core.add(layers.Dense(128, activation='sigmoid', name='fc_dense_3'))
        
        return core
        
    def createModel(self, name, input_shape, lr, margin, core_path=None):
        left = layers.Input(input_shape)
        right = layers.Input(input_shape)
        
        left_encoded = self.core(left)
        right_encoded = self.core(right)
        
        dist = layers.Lambda(lambda x: self.euclidean_distance(x))
        output = dist([left_encoded, right_encoded])
        
        siamese = Model(inputs=[left, right], outputs=output, name=name)
        
        optimizer = Adam(lr=lr)
        siamese.compile(loss=self.contrastive_loss(margin), optimizer=optimizer)
        
        return siamese
        
    # def cosine(self, inputs):
    #   u, v = inputs
    #   return cosine_similarity(u, v)
      
    # def contrastive_loss(self, margin):
    #     def loss(y, d):
    #         positive = (1 - y) * d
    #         negative = y * (margin - d)
    #         # K.print_tensor(positive, message='POSITVE')
    #         # K.print_tensor(negative, message='NEGATIVE')
    #         return K.mean(positive + negative)
    #     return loss 

    def euclidean_distance(self, inputs):
        u, v = inputs
        return K.sqrt(K.maximum(K.sum(K.square(u - v), axis=1, keepdims=True), K.epsilon()))

    def contrastive_loss(self, margin):
        def loss(y, d):
            positive = (1 - y) * K.square(d)
            negative = y * K.square(K.maximum(margin - d, 0.))
            # K.print_tensor(positive, message='POSITVE')
            # K.print_tensor(negative, message='NEGATIVE')
            return K.mean(positive + negative)
        return loss
