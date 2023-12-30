import os
import math
import numpy as np
import datetime as dt
import tensorflow as tf
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM, Lambda, Layer, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import Adam,SGD
from tensorflow.keras.optimizers import Adam,SGD
import keras.backend as K

class CustomLayer(Layer):     
	def __init__(self, **kwargs):         
		super(CustomLayer, self).__init__(**kwargs)      
		
	def build(self, input_shape):         
		super(CustomLayer, self).build(input_shape)      
		
	def call(self, x):  
		def process(sample):
			p1 = sample[0] # [mu1 - mu5]
			p2 = K.exp(sample[1]) # [s1 - s5]
			return tf.concat([[p1],[p2]],axis=0)
		out = tf.map_fn(process, x, dtype=tf.float32)
		return out
	
	def compute_output_shape(self, input_shape):         
		return (input_shape[0],2,5)

class Model():
	"""LSTM 模型"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath,custom_objects={'CustomLayer': CustomLayer},compile=False)
		# return self.model

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		# self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		# 单独添加一层dense和Lambda，保证输出的一个维度非负
		self.model.add(CustomLayer())

		print('[Model] Model Compiled')
		timer.stop()
		
		return self.model
	

	def train(self, x, y, epochs, batch_size, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def predict_point_by_point(self, data,debug=False):
		if debug == False:
			print('[Model] Predicting Point-by-Point...')
			predicted = self.model.predict(data)
			predicted = np.reshape(predicted, (predicted.size,))
		else:
			print('[Model] Predicting Point-by-Point...')
			print (np.array(data).shape)
			predicted = self.model.predict(data)
			print (np.array(predicted).shape)
			predicted = np.reshape(predicted, (predicted.size,))
			print (np.array(predicted).shape)
		return predicted
	
	def predict_sequences_multiple(self, data, window_size, prediction_len,debug=False):
		if debug == False:
			print('[Model] Predicting Sequences Multiple...')
			prediction_seqs = []
			for i in range(int(len(data)/prediction_len)):
				curr_frame = data[i*prediction_len]
				predicted = []
				for j in range(prediction_len):
					predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
					curr_frame = curr_frame[1:]
					curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
				prediction_seqs.append(predicted)
			return prediction_seqs
		else :
			print('[Model] Predicting Sequences Multiple...')
			prediction_seqs = []
			for i in range(int(len(data)/prediction_len)):
				print (data.shape)
				curr_frame = data[i*prediction_len]
				print (curr_frame)
				predicted = []
				for j in range(prediction_len):
					predict_result = self.model.predict(curr_frame[newaxis,:,:])
					print (predict_result)
					final_result = predict_result[0,0]
					predicted.append(final_result)
					curr_frame = curr_frame[1:]
					print (curr_frame)
					curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
					print (curr_frame)
				prediction_seqs.append(predicted)
		

	def predict_sequence_full(self, data, window_size):
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted


class Bayesian_LSTM(Model):
	def __init__(self,):
		super(Model, self).__init__()
		self.model = Sequential()

	def build_model(self, configs): 		
		timer = Timer() 		
		timer.start()  		
		for layer in configs['model']['layers']: 			
			neurons = layer['neurons'] if 'neurons' in layer else None 			
			dropout_rate = layer['rate'] if 'rate' in layer else None 			
			activation = layer['activation'] if 'activation' in layer else None 			
			return_seq = layer['return_seq'] if 'return_seq' in layer else None 			
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None 			
			input_dim = layer['input_dim'] if 'input_dim' in layer else None  			
			if layer['type'] == 'dense': 				
				self.model.add(Dense(neurons, activation=activation)) 			
			if layer['type'] == 'lstm': 				
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq)) 			
			if layer['type'] == 'dropout': 				
				self.model.add(Dropout(dropout_rate))  	
		
		# [[mu1,mu2,mu3,mu4,mu5], [s1,s2,s3,s4,s5]]
		self.model.add(Reshape((2,5)))

		# 单独添加一层dense和Lambda，保证输出的一个维度非负 		
		self.model.add(CustomLayer())  		
		print('[Model] Model Compiled') 		
		timer.stop() 		 		
		return self.model
	
	def load_model(self, filepath):
		super().load_model(filepath)

	def bayesian_train(self, train_dataset, epochs,Num_sample, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')

		optimizer = Adam()

		fname = f"{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}-e{str(epochs)}"

		# save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		save_fname = os.path.join(save_dir, '%s.h5' % (fname))
		print(F"Total epochs:{epochs}")
		# 自定义训练过程  
		for e in range(epochs):
			print("\nStart epoch %d" % (e+1))
			print(f"Total batch:{len(train_dataset)}")
			for batch_idx, (train_x, train_y) in enumerate(train_dataset):
				with tf.GradientTape() as tape:
					out = self.model(train_x ,training=True)


					mu,sigma = tf.split(out, 2, axis=1) # mu: [[[mu1 ... mu5]]]

					# print(mu.shape,sigma.shape,train_y.shape)	
					mu = tf.squeeze(mu,axis=1) # mu: [[mu1,...,mu5]]
					sigma = tf.squeeze(sigma,axis=1)					
					# print(mu.shape,sigma.shape)

					sample_total = tf.zeros((Num_sample, tf.shape(train_x)[0], tf.shape(train_y)[1]))
					# print(sample_total.shape)
					# 对均值为mu,方差为sigma^2的正态分布抽样
					for t in tf.range(Num_sample):
						epsilon = tf.random.normal(tf.shape(sigma))
						sample = mu + tf.multiply(sigma, epsilon)
						sample_total = tf.tensor_scatter_nd_update(sample_total, [[t]], [sample])
						# print(sample_total[t] == sample)

					sample_ave = tf.reduce_mean(sample_total, axis=0) 
					# print(sample_ave[0][0] == 0.5*(sample_total[0][0][0]+sample_total[1][0][0]), sample_ave[1][1] == 0.5*(sample_total[0][1][1]+sample_total[1][1][1]))

					loss_val = self.loss_fun(sample_ave, train_y, sigma)

				grads = tape.gradient(loss_val, self.model.trainable_weights)

				optimizer.apply_gradients(zip(grads,self.model.trainable_weights))

				print(f"Current batch index:{batch_idx}, Curret loss:{loss_val}", end='\r')

		self.model.save(save_fname)  		
		print('[Model] Training Completed. Model saved as %s' % save_fname) 		
		timer.stop()
		return fname
	
	def loss_fun(self, sample_y, train_y, sigma): 
		# print(sample_y.dtype)
		# print(train_y.dtype)
		# print(sigma.dtype)

		train_y = tf.cast(train_y, dtype=tf.float32)

		rows, cols = train_y.shape 		
	
		square_loss = tf.math.square((train_y - sample_y) / sigma) 
		log_loss = 2 * tf.math.log(sigma) 
		sample_loss = square_loss + log_loss  
		loss_val = tf.reduce_sum(sample_loss, axis=1) / (2 * cols)   
		return tf.reduce_mean(loss_val)	


	def bayesian_predict(self, testx, window_size, prediction_len,plot=False):
		"""
		"""
		if plot == False:
			print("[Model] Predicting By Bayesian-LSTM ...")
			prediction_seqs = []
			for i in range(int(len(testx)/prediction_len)):
				curr_frame = testx[i*prediction_len]
				predicted = []
				for j in range(prediction_len):
					mu, sigma = self.model.predict(curr_frame[newaxis,:,:])[0]
					predicted.append(mu)
					curr_frame = curr_frame[1:]
					curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
				prediction_seqs.append(predicted)
			return prediction_seqs
		else:
			# plot == true，为了画图，取predicted中的1维
			print("[Model] Plotting Prediction ...")
			prediction_seqs = []
			for i in range(int(len(testx)/prediction_len)):
				curr_frame = testx[i*prediction_len]
				predicted = []
				for j in range(prediction_len):
					mu, sigma = self.model.predict(curr_frame[newaxis,:,:])[0]
					predicted.append(mu[0]) # 取第一个画图
					curr_frame = curr_frame[1:]
					curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
				prediction_seqs.append(predicted)
			return prediction_seqs
