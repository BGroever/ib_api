from matplotlib import pyplot as plt
from IPython.display import clear_output
import tensorflow.keras as keras


class PlotProgress(keras.callbacks.Callback):
	def __init__(self):
		self.i = 0
		self.x = []
		
		self.losses = []
		self.val_losses = []
		self.acc = []
		self.val_acc = []
		
	def on_train_begin(self, logs={}):        
		self.figures = []
		plt.figure()
		self.figures.append(plt.gcf().number)
		
		plt.figure()
		self.figures.append(plt.gcf().number)
		
		#self.fig2 = plt.figure()
		
		self.logs = []

	def on_epoch_end(self, epoch, logs={}):
		
		self.logs.append(logs)
		self.x.append(self.i)
		
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		
		self.acc.append(logs.get('accuracy'))
		self.val_acc.append(logs.get('val_accuracy'))
		self.i += 1
		
		clear_output(wait=True)
		#plt.subplot(121)
		plt.figure(self.figures[0]);
		plt.plot(self.x, self.acc, label="accuracy");
		plt.plot(self.x, self.val_acc, label="val_accuracy");
		plt.legend();
		plt.show();
		#plt.subplot(122)
		plt.figure(self.figures[1]);
		plt.plot(self.x, self.losses, label="loss");
		plt.plot(self.x, self.val_losses, label="val_loss");
		plt.legend();
		plt.show();
		print(logs.get('loss'), logs.get('val_loss'),logs.get('accuracy'),logs.get('val_accuracy'))


# ************************************************************************** #
#																			 #
# 					Data generator 											 #
# 																			 #
# ************************************************************************** #		

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np

class DataGenerator(Sequence):
 
	"""
		Generates data for Keras
		Sequence based data generator.

	"""
	
	def __init__(self, df, 
				 look_back=100,
				 batch_size=600, 
				 shuffle=True, 
				 training=True,
				 categorical=True,
				 patch_to_4D=False,
				 cut: float = 0.1):
		
		"""Initialization
		:param df: dataframe
		:param look_back: int     size of the look back
		:param batch_size: int    size of the batch
		:param shuffle: bool      shuffle data after each epoch?
		:param training: bool     traning mode, enables resampling
		:param categorical: bool  one-hot incoding
		:param patch_to_4D: bool  patch to 4D for Conv2D input instad of Conv1D input
		"""
		
		self.df     = df
		self.y      = df['ZZ'].to_numpy()
		self.target = df['target'].to_numpy()
		self.z      = df[['time','open','high','low','close']].to_numpy()

		self.look_back   = look_back
		self.batch_size  = batch_size
		self.shuffle     = shuffle
		self.training    = training
		self.categorical = categorical
		self.patch_to_4D = patch_to_4D
		
		self.classes = self.target[-(self.y.shape[0]-look_back+1):]
		self.z       = self.z[-(self.y.shape[0]-look_back+1):]

	   #****************************************************************************
	   # Determine the indicies 
	   #****************************************************************************

		if self.training:
			holds = []
			buys  = []
			sells = []

			for i in range(self.look_back-1, self.y.shape[0]):
				if self.y[i]==0:
					holds.append(i)
				if self.y[i]==1:
					sells.append(i)
				if self.y[i]==2:
					buys.append(i)

			self.holds = np.array(holds)
			self.buys  = np.array(buys)
			self.sells = np.array(sells)

			self.ids = np.arange(self.holds.shape[0]).tolist()
			
		else: 
			self.shuffle = False
			self.ids = np.arange(self.look_back-1,self.y.shape[0])

		self.on_epoch_end()


	   #****************************************************************************
	   # Sort the columns into feature dictionary
	   #****************************************************************************

		features = df.columns.to_list().copy()

		for i in range(len(features)):
			features[i] = features[i].split('.')[0]
			
		features = np.unique(features).tolist()

		for item in ['close', 'high', 'low', 'open', 'time', 'ZZ', 'target']:
			if item in features:
				features.remove(item)

		features = {feature: [] for feature in features}

		for item in df.columns.copy():
			feature = item.split('.')[0]
			if features.get(feature)!= None:
				features.get(feature).append(item)
				
		for feature in features.keys():
			features[feature] = df[features[feature]].to_numpy()

		self.features = features
		self.keys     = None

	   #****************************************************************************
	   # Print values
	   #****************************************************************************

		print(self.z.shape, self.classes.shape)

	def on_epoch_end(self):
		"""
			Updates indexes after each epoch
		"""
		if self.shuffle == True:
			np.random.shuffle(self.ids)

	def counts(self, keys: list = None):
		"""
			Accepts key list and return a dictionary of counts for each name
		"""

		if keys == None or keys == []: 
			keys = list(self.features.keys())
		self.keys = keys

		counts = {}
		for key in self.keys:
			if key == 'last_kind':
				counts[key] = 1
				continue
			if isinstance(key, str):
				counts[key] = self.features[key].shape[-1]
			if isinstance(key, list):
				string = key[0]
				count = self.features[key[0]].shape[-1]
				for item in key[1:]:
					string += '-' + item
					count += self.features[item].shape[-1]
				counts[string] = count

		counts['trend'] += 1
		return counts

	def __len__(self):
		"""
			Denotes the number of batches per epoch
			:return: number of batches per epoch
		"""
		if self.training:
			return int(np.floor(len(self.ids) / self.batch_size))
		else:
			return int(np.ceil(len(self.ids) / self.batch_size))


	def __getitem__(self, index):
		"""
			Generate one batch of data
			:param index: index of the batch
			:return: X and y 
		"""

	   #*********************************************************************************
	   # Determine ids included in the batch
	   #*********************************************************************************
		
		if self.training:
			hold_ids = self.holds[np.array(self.ids[index * self.batch_size:(index + 1) * self.batch_size])]
			buy_ids  = np.random.choice(self.buys, hold_ids.shape, replace=True)
			sell_ids = np.random.choice(self.sells, hold_ids.shape, replace=True)

			assert(hold_ids.shape == buy_ids.shape)
			assert(buy_ids.shape  == sell_ids.shape)
			ids = np.concatenate([hold_ids,sell_ids,buy_ids])

		else:
			ids = np.array(self.ids[index * self.batch_size:(index + 1) * self.batch_size])


	   #*********************************************************************************
	   # Define target mask function for 'ZZ trend' feature
	   #*********************************************************************************


		def mask_target(ZZ: np.ndarray, trend: np.ndarray):
			"""
				Mask the target to eliminate data unknown at time the end of ZZ
				Returns a copy of ZZ and the kind of the last extrema {2: 'MIN', 1: 'MAX'}
			"""
			ZZ = ZZ.copy()

			# find location when last extrema was identified
			flips = np.where(trend == 0 if trend[-1]==1 else trend==1)[0]
			if flips.shape[0] == 0:
				return np.ones(ZZ.shape) * 3, -1

			# find last extrema 
			extremas = np.where(ZZ != 0)[0]
			possible_extremas = np.where(extremas < int(flips[-1]))[0]
			if possible_extremas.shape[0] == 0:
				return np.ones(ZZ.shape) * 3, -1

			mask_until = int(extremas[possible_extremas[-1]])
			ZZ[mask_until+1:] = 3
			return ZZ, ZZ[mask_until]

	   #*********************************************************************************
	   # Construct an array for each feature with shape (None, look_back, num_feature)
	   #*********************************************************************************

		X = {feature: np.zeros((ids.shape[0], self.look_back, self.features[feature].shape[-1])) for feature in self.features.keys()}
		y = self.target[ids]

		for feature in self.features:
			count = 0
			for index in ids:
				X[feature][count] = self.features[feature][index-self.look_back+1:index+1,:]
				count += 1

	   #*********************************************************************************
	   # Add masked ZZ values to 'trend' feature
	   # Add last_kind to X
	   #*********************************************************************************

		ZZ_val = np.zeros(X['trend'].shape)
		last_kind = np.zeros(ids.shape)
			
		count = 0     
		for index in ids:
			ZZ,extrema = mask_target(self.y[index-self.look_back+1:index+1], X['trend'][count])
			ZZ_val[count]    = ZZ.reshape(*ZZ.shape,1)
			last_kind[count] = extrema
			count += 1

		X['last_kind']  = last_kind.reshape(*last_kind.shape,1)
		X['trend']      = np.concatenate((X['trend'], ZZ_val), axis=-1)

	   #********************************************************************************
	   # Combine features according to keys
	   #********************************************************************************

		if self.keys != None:
			X_new = {}
			for key in self.keys:
				if isinstance(key, str):
					X_new[key] = X[key]
				if isinstance(key, list):
					string = key[0]
					array = X[key[0]]
					for item in key[1:]:
						string += '-' + item
						array = np.concatenate((array, X[item]), axis=-1)
					X_new[string] = array
			X = X_new

	   #********************************************************************************
	   # Transform data in to proper output shape
	   #********************************************************************************

	    # Convert to one-hot encoding
		# (0,1,2,1) -> ((1,0,0),(0,1,0),(0,0,1),(0,1,0))
		if self.categorical:
			y = to_categorical(y)

		# (None, batch_size, timesteps, num_features) -> (None, batch_size, timesteps, num_features, 1)
		if self.patch_to_4D:
			pass

		return X, y





