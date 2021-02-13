from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def Conv1D():
    core = Sequential()
    # core.add(layers.LayerNormalization(axis=2))
    core.add(layers.TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='relu'), name='td_conv1d_1'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    core.add(layers.TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv1d_2'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    core.add(layers.TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv1d_3'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    core.add(layers.TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv1d_4'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_4'))
    core.add(layers.TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv1d_5'))
    core.add(layers.GlobalMaxPooling2D(name='global_maxpool2d'))
    core.add(layers.Flatten(name='flatten'))
    
    return core

def Conv2D():
    core = Sequential()
    # core.add(layers.LayerNormalization(axis=2))
    core.add(layers.Conv2D(8, kernel_size=(7, 7), activation='relu', name='conv2d_1'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    core.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu', name='conv2d_2'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    core.add(layers.Conv2D(24, kernel_size=(3, 3), activation='relu', name='conv2d_3'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    core.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2d_4'))
    core.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_4'))
    core.add(layers.Conv2D(48, kernel_size=(3, 3), activation='relu', name='conv2d_5'))
    core.add(layers.Flatten(name='flatten'))
    
    return core

def LSTM():
    core = Sequential()
    # core.add(layers.LayerNormalization(axis=2))
    core.add(layers.TimeDistributed(layers.Reshape((-1,)), name='td_reshape'))
    core.add(layers.TimeDistributed(layers.Dense(64, activation='relu'), name='td_dense'))
    core.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True), name='bi_lstm'))
    core.add(layers.Dense(64, activation='relu', name='dense'))
    core.add(layers.MaxPooling1D(name='maxpool1d'))
    core.add(layers.Flatten(name='flatten'))
    
    return core

def createModel(input_shape, N_CLASSES=100, name='conv_1d'):
    i = layers.Input(input_shape)
    if name == 'conv_1d':
        core = Conv1D()
    elif name == 'conv_2d':
        core = Conv2D()
    else:
        core = LSTM()
        
    x = core(i)
    x = layers.Dense(1024, activation='relu', name='dense_1')(x)
    x = layers.Dense(512, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    x = layers.Dense(256, activation='relu', name='dense_3')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)
    model = Model(inputs=i, outputs=o, name=name)
    model.compile(optimizer=Adam(lr=0.0006), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
