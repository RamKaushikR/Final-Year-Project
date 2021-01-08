from tensorflow.keras import layers
from tensorflow.keras.models import Model
from Contrastive.siamese import Siamese as Contrastive
from Triplet.siamese import Siamese as Triplet


CONTRASTIVE_LSTM = 'Contrastive/Model/LSTM/lstm_50.h5'
CONTRASTIVE_CONV1D = 'Contrastive/Model/Conv1D/conv1d_50.h5'
CONTRASTIVE_CONV2D = 'Contrastive/Model/Conv2D/conv2d_50.h5'

TRIPLET_LSTM = 'Triplet/Model/LSTM/lstm_50.h5'
TRIPLET_CONV1D = 'Triplet/Model/Conv1D/conv1d_50.h5'
TRIPLET_CONV2D = 'Triplet/Model/Conv2D/conv2d_50.h5'

INPUT_SHAPE = (300, 128, 1)

def createModel(name='contrastive_lstm'):
	if name == 'contrastive_lstm':
		siamese = Contrastive(INPUT_SHAPE, name='lstm', model_path=CONTRASTIVE_LSTM, margin=0.5)
			
	elif name == 'contrastive_conv1d':
		siamese = Contrastive(INPUT_SHAPE, name='conv_1d', model_path=CONTRASTIVE_CONV1D, margin=0.5)
			
	elif name == 'contrastive_conv2d':
		siamese = Contrastive(INPUT_SHAPE, name='conv_2d', model_path=CONTRASTIVE_CONV2D, margin=0.5)
			
	elif name == 'triplet_lstm':
		siamese = Triplet(INPUT_SHAPE, name='lstm', model_path=TRIPLET_LSTM, margin=1)
			
	elif name == 'triplet_conv1d':
		siamese = Triplet(INPUT_SHAPE, name='conv_1d', model_path=TRIPLET_CONV1D, margin=1)
			
	elif name == 'triplet_conv2d':
		siamese = Triplet(INPUT_SHAPE, name='conv_2d', model_path=TRIPLET_CONV2D, margin=1)
		
	model = siamese.model.get_layer('sequential')
	return model
