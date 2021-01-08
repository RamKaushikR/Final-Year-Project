import os
import pickle
import librosa
import numpy as np
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import backend as K


SPEAKERS_PATH = 'Speakers/'
PREDICT_PATH = 'Predict/'
SR = 16000
TIME = 3 * SR
AUDIO_SHAPE = (TIME, 1)
SPEC_SHAPE = (300, 128, 1)

def spectrogram(audio):
	layer = get_melspectrogram_layer(input_shape=AUDIO_SHAPE,
                                     n_mels=128, pad_end=True,
                                     n_fft=512, win_length=400,
                                     hop_length=160, sample_rate=SR,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_last'
                                    )
                                    
	audio = np.expand_dims(audio, axis=1)
	audio = np.reshape(audio, (1, audio.shape[0], audio.shape[1]))
	spec = layer(audio)
	return spec

def add(speaker):
	a, r = librosa.load(SPEAKERS_PATH + speaker + '.wav')
	a = librosa.resample(a, r, SR)
	if a.shape[0] < TIME:
		return -1
		
	i = np.random.randint(0, a.shape[0] - TIME)
	audio = a[i:i+TIME]
	
	spec = spectrogram(audio)
	spec = np.reshape(spec, (spec.shape[1], spec.shape[2]))
	
	with open(SPEAKERS_PATH + speaker + '.dat', 'wb') as f:
		pickle.dump(spec, f)
	os.remove(SPEAKERS_PATH + speaker + '.wav')
	
	return 1
	
def remove(speaker):
	os.remove(SPEAKERS_PATH + speaker + '.dat')
	return 1
	
def euclideanDistance(inputs):
	u, v = inputs
	return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))
	
def predict(model):
	s = os.listdir(SPEAKERS_PATH)
	p = os.listdir(PREDICT_PATH)
	
	speakers = []
	for _ in s:
		spec = pickle.load(open(SPEAKERS_PATH + _, 'rb'))
		spec = np.reshape(spec, SPEC_SHAPE)
		speakers.append(abs(spec))
	speakers = np.array(speakers)
	
	a, r = librosa.load(PREDICT_PATH + p[0])
	a = librosa.resample(a, r, SR)
	if a.shape[0] < TIME:
		return -1
		
	i = np.random.randint(0, a.shape[0] - TIME)
	audio = a[i:i+TIME]
	
	predict_spec = abs(spectrogram(audio))
	predict_spec = np.reshape(predict_spec, (1, 300, 128, 1))
	
	y_true = model.predict(speakers)
	y_pred = model.predict(predict_spec)
    
# 	print("y_true:", y_true, y_true.shape)
# 	print("y_pred:", y_pred, y_pred.shape)
	dist = euclideanDistance((y_true, y_pred))
	index = np.argmin(dist)
	print(dist, s)
	
	return s[index][:-4]
