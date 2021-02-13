import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend as K
from .model import Conv1D, Conv2D, LSTM


class Siamese(object):
    def __init__(self, input_shape, lr=0.001, core_path=None, model_path=None, name='conv_1d', margin=1, encoding_dim=128):
        self.w = input_shape[0]
        self.h = input_shape[1]
        
        self.core = self.createCore(name, encoding_dim)
            
        self.model = self.createModel(name, input_shape, lr, margin, encoding_dim=encoding_dim)
        
        if model_path:
            self.model.load_weights(model_path)
        if core_path:
            self.model.get_layer('sequential').load_weights(core_path, by_name=True)
            for layer in self.model.get_layer('sequential').layers[:-3]:
                layer.trainable = True
    
    def createCore(self, name='conv_1d', encoding_dim=128):
        if name == 'conv_1d':
            core = Conv1D()
            
        elif name == 'conv_2d':
            core = Conv2D()
            
        else:
            core = LSTM()
            
        core.add(layers.Dense(256, activation='relu', activity_regularizer=l2(0.001), name='fc_dense_1'))
        core.add(layers.Dense(192, activation='sigmoid', name='fc_dense_2'))
        core.add(layers.Dense(encoding_dim, activation='sigmoid', name='fc_dense_3'))
        
        return core
        
    def createModel(self, name, input_shape, lr, margin, encoding_dim=128):
        anchor = layers.Input(input_shape)
        positive = layers.Input(input_shape)
        negative = layers.Input(input_shape)     
        
        anchor_encoded = self.core(anchor)
        positive_encoded = self.core(positive)
        negative_encoded = self.core(negative)
        
        output = layers.concatenate([anchor_encoded, positive_encoded, negative_encoded], axis=1, name='concatenate')
        siamese = Model(inputs=[anchor, positive, negative], outputs=output, name='output')
        
        optimizer = Adam(lr=lr)
        siamese.compile(loss=self.triplet_loss(margin, encoding_dim), optimizer=optimizer)
        
        return siamese
        
    def euclidean_distance(self, inputs):
        u, v = inputs
        return K.sqrt(K.maximum(K.sum(K.square(u - v), axis=1, keepdims=True), K.epsilon()))
        
    def triplet_loss(self, margin, encoding_dim):
        def loss(y_true, y_pred):
            anchor, positive, negative = y_pred[:, :encoding_dim], \
                                         y_pred[:, encoding_dim:2*encoding_dim], y_pred[:, 2*encoding_dim:]
            dist_pos = self.euclidean_distance((anchor, positive))
            dist_neg = self.euclidean_distance((anchor, negative))
            # K.print_tensor(dist_pos, message='POSITIVE')
            # K.print_tensor(dist_neg, message='NEGATIVE')
            return K.mean(K.maximum(dist_pos - dist_neg + margin, 0.))
        return loss
        
    # def triplet_loss(alpha, encoding_dim):
	#     def loss(y_true, y_pred):
	#         anchor, positive, negative = y_pred[:, :encoding_dim], \
	#                                      y_pred[:, encoding_dim:2*encoding_dim], y_pred[:, 2*encoding_dim:]
	#         dist_pos = cosine_similarity(anchor, positive)
	#         dist_neg = cosine_similarity(anchor, negative)
	#         return K.clip(dist_pos - dist_neg + alpha, 0., None)
	#     return loss
