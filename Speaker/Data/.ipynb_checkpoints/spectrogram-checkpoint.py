import numpy as np
from kapre.composed import get_melspectrogram_layer


class Spectrogram:
    def __init__(self, input_shape, sr):
        self.layer = get_melspectrogram_layer(input_shape=input_shape,
                                              n_mels=128, pad_end=True,
                                              n_fft=512, win_length=400,
                                              hop_length=160, sample_rate=sr,
                                              return_decibel=True,
                                              input_data_format='channels_last',
                                              output_data_format='channels_last'
                                             )
        
    def spectrogram(self, audio):
        audio = np.reshape(audio, (1, audio.shape[0], audio.shape[1]))
        spec = self.layer(audio)
        spec = np.reshape(spec, (spec.shape[1], spec.shape[2]))
        return spec
