import os
import pickle
import librosa
import numpy as np
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import backend as K
import speech_recognition as sr


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
	
def speechToText(recognizer, path, audio_text):
	a = sr.AudioFile(path)
	text = ''
	
	with a as source:
		audio = recognizer.record(source)
		text = recognizer.recognize_google(audio)
		
	text = text.lower()
	audio_text = audio_text.lower()
	print('Received text is: ' + audio_text)
	print('Speech to text is: ' + text)
	
	pred_dict, audio_dict = {}, {}
	
	for word in list(text.split(' ')):
		pred_dict[word] = pred_dict[word] + 1 if word in pred_dict else 1
	for word in list(audio_text.split(' ')):
		audio_dict[word] = audio_dict[word] + 1 if word in audio_dict else 1
		
	count = 0
	for word in pred_dict:
		if word in audio_dict:
			count += min(audio_dict[word], pred_dict[word])
				
	return count / sum(audio_dict.values())

def add(speaker, recognizer, audio_text, dependent):
	if speaker + '.dat' in os.listdir(SPEAKERS_PATH):
		os.remove(SPEAKERS_PATH+speaker+'.wav')
		return speaker + ' has already been enrolled'

	if dependent == 'True':
		text = speechToText(recognizer, SPEAKERS_PATH+speaker+'.wav', audio_text)
		if text < 0.70:
			os.remove(SPEAKERS_PATH+speaker+'.wav')
			return 'Text match unsuccessful'

	a, r = librosa.load(SPEAKERS_PATH+speaker+'.wav')
	a = librosa.resample(a, r, SR)
	if a.shape[0] < TIME:
		os.remove(SPEAKERS_PATH+speaker+'.wav')
		return 'Invalid File. Talk a little slower'
		
	i = (a.shape[0] - TIME) // 2#np.random.randint(0, a.shape[0] - TIME)
	audio = a[i:i+TIME]
	
	spec = abs(spectrogram(audio))
	spec = np.reshape(spec, (spec.shape[1], spec.shape[2]))
	
	with open(SPEAKERS_PATH+speaker+'.dat', 'wb') as f:
		pickle.dump(spec, f)
	os.remove(SPEAKERS_PATH+speaker+'.wav')
	
	return speaker + ' added successfully'
	
def remove(speaker):
	if speaker + '.dat' in os.listdir(SPEAKERS_PATH):
		os.remove(SPEAKERS_PATH+speaker+'.dat')
		return speaker + ' removed successfully'
		
	return speaker + ' has not been enrolled'
	
def euclideanDistance(inputs):
	u, v = inputs
	return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))
	
def predict(audio_text, model, threshold, recognizer, dependent):
	result = {'text': '', 'speaker': ''}
	s = os.listdir(SPEAKERS_PATH)
	p = os.listdir(PREDICT_PATH)
	
	if dependent == 'True':
		text = speechToText(recognizer, PREDICT_PATH+p[0], audio_text)
		if text < 0.6:
			result['text'] = 'Text match unsuccessful'
		else:
			result['text'] = 'Text match successful'

	speakers = []
	for _ in s:
		spec = pickle.load(open(SPEAKERS_PATH+_, 'rb'))
		spec = np.reshape(spec, SPEC_SHAPE)
		speakers.append(abs(spec))
	speakers = np.array(speakers)
	
	a, r = librosa.load(PREDICT_PATH+p[0])
	a = librosa.resample(a, r, SR)
	if a.shape[0] < TIME:
		result['text'] = 'Invalid File. Talk a little slower'
		result['speaker'] = ''
		return result
		
	i = (a.shape[0] - TIME) // 2#np.random.randint(0, a.shape[0] - TIME)
	audio = a[i:i+TIME]
	
	predict_spec = abs(spectrogram(audio))
	predict_spec = np.reshape(predict_spec, (1, 300, 128, 1))
	
	y_true = model.predict(speakers)
	y_pred = model.predict(predict_spec)
    
	dist = euclideanDistance((y_true, y_pred))
	index = np.argmin(dist)
	print(dist, s)
	
	if dist[index] >= threshold:
		result['speaker'] = 'Speaker Not Found'
	else:
		result['speaker'] = s[index][:-4] + ' is the identified speaker'
		
	return result
