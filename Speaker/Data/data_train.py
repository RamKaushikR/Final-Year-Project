import sys
import os
import warnings
import pickle
import numpy as np
import librosa
from spectrogram import Spectrogram


warnings.filterwarnings('ignore')
np.random.seed(60)
DATA_PATH = '../Dataset/train'

def loadFile(path, sr=16000, time=3.0):
    spectrogram = Spectrogram(input_shape=(int(time*sr), 1), sr=sr)
    X = []
    curr = 0 # indicates 1st and last file of speaker
    speaker_dict = {}
    
    print('Total speakers:', len(os.listdir(path)))
    c = 0

    for speaker in os.listdir(path):
        if c % 10 == 0:
            print('Speakers done:', c)
        c += 1
        print('Loading speaker: ' + speaker)
        speaker_path = os.path.join(path, speaker)
        speaker_dict[speaker] = [curr, None]
        X_speaker = []

        for file_path in os.listdir(speaker_path):
            files = os.path.join(speaker_path, file_path)
            audio, rate = librosa.load(files, sr=sr)
            audio = np.expand_dims(audio, axis=1)

            spec = spectrogram.spectrogram(audio)
            X_speaker.append(spec)
            curr += 1

        speaker_dict[speaker][1] = curr - 1
        X.append(X_speaker)

    return X, speaker_dict


X_train, s_train = loadFile(DATA_PATH)
with open('Data/train.pickle', 'wb') as f:
    pickle.dump((X_train, s_train), f)
print('Train dataset done')
del X_train, s_train
