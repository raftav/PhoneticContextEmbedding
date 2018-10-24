from __future__ import division
import tensorflow as tf
import sys

class Model(object):

	def __init__(self,phone_features,acoustic_features,labels,seq_length,config,is_training):

		if is_training:
			batch_size=config.batch_size
			dropout_keep_prob=0.8
		else:
			batch_size=1
			dropout_keep_prob=1.0

		global_step = tf.Variable(0, trainable=False)
		self._global_step=global_step

		with tf.variable_scope('phonetic_net'):
			with tf.variable_scope('embedding'):
				with tf.variable_scope('forward'):
					lstm_cell_forward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.embedding_size,
												forget_bias=1.0,
												activation=tf.nn.relu,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=1.0)
				with tf.variable_scope('backward'):
					lstm_cell_backward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.embedding_size,
												forget_bias=1.0,
												activation=tf.nn.relu,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=1.0)

				with tf.variable_scope('embedding_rnn'):
					embedding_output, embedding_states = tf.nn.bidirectional_dynamic_rnn(
														cell_fw=lstm_cell_forward,
														cell_bw=lstm_cell_backward,
														inputs=phone_features,
														sequence_length=seq_length,
														initial_state_fw=lstm_cell_forward.zero_state(batch_size,dtype=tf.float32),
														initial_state_bw=lstm_cell_backward.zero_state(batch_size,dtype=tf.float32),
														dtype=tf.float32)

					phone_embedding = tf.concat(embedding_output,2,name='recurrent_embedding')

			# lstm cells definition
			with tf.variable_scope('forward'):

				forward_cells = []
				forward_cells_init = []
				for i in range(config.phonetic_num_layers):
					with tf.variable_scope('layer_{:d}'.format(i)):
						lstm_cell_forward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.phonetic_num_neurons,
												forget_bias=1.0,
												activation=tf.nn.relu,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=dropout_keep_prob)
						forward_cells.append(lstm_cell_forward)
						forward_cells_init.append(lstm_cell_forward.zero_state(batch_size,dtype=tf.float32))

			with tf.variable_scope('backward'):

				backward_cells = []
				backward_cells_init = []
				for i in range(config.phonetic_num_layers):
					with tf.variable_scope('layer_{:d}'.format(i)):
						lstm_cell_backward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.phonetic_num_neurons,
												forget_bias=1.0,
												activation=tf.nn.relu,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=dropout_keep_prob)
						backward_cells.append(lstm_cell_backward)
						backward_cells_init.append(lstm_cell_backward.zero_state(batch_size,dtype=tf.float32))

			with tf.variable_scope('RNN'):
				rnn_outputs, output_state_fw , output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
												cells_fw=forward_cells,
												cells_bw=backward_cells,
												inputs=phone_embedding,
												initial_states_fw=forward_cells_init,
												initial_states_bw=backward_cells_init,
												dtype=tf.float32,
												sequence_length=seq_length)


			with tf.variable_scope('output'):
				output_weights = tf.get_variable('phone_weights',[2*config.phonetic_num_neurons,config.acoustic_feat_dimension],dtype=tf.float32,
										initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
				output_biases = tf.get_variable('phone_biases',shape=[config.acoustic_feat_dimension],dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

				rnn_outputs = tf.reshape(rnn_outputs,[-1,2*config.phonetic_num_neurons])	
				
				acoustic_reconstruction = tf.matmul(rnn_outputs,output_weights) + output_biases
			
				acoustic_reconstruction = tf.reshape(acoustic_reconstruction,[batch_size,-1,config.acoustic_feat_dimension])

			with tf.name_scope('reconstruction_loss'):
				diff = tf.squared_difference(acoustic_reconstruction,acoustic_features)
				self._reconstruction_loss = tf.reduce_mean( tf.reduce_sum(diff,axis=2) )

		with tf.variable_scope('acoustic_net'):
			with tf.variable_scope('forward'):
				forward_cells = []
				forward_cells_init = []
				for i in range(config.acoustic_num_layers):
					with tf.variable_scope('layer_{:d}'.format(i)):
						lstm_cell_forward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.acoustic_num_neurons,
												forget_bias=1.0,
												activation=tf.tanh,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=dropout_keep_prob)
						forward_cells.append(lstm_cell_forward)
						forward_cells_init.append(lstm_cell_forward.zero_state(batch_size,dtype=tf.float32))

			with tf.variable_scope('backward'):

				backward_cells = []
				backward_cells_init = []
				for i in range(config.acoustic_num_layers):
					with tf.variable_scope('layer_{:d}'.format(i)):
						lstm_cell_backward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.acoustic_num_neurons,
												forget_bias=1.0,
												activation=tf.tanh,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=dropout_keep_prob)
						backward_cells.append(lstm_cell_backward)
						backward_cells_init.append(lstm_cell_backward.zero_state(batch_size,dtype=tf.float32))

			with tf.variable_scope('RNN'):
				rnn_outputs, output_state_fw , output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
												cells_fw=forward_cells,
												cells_bw=backward_cells,
												inputs=acoustic_features,
												initial_states_fw=forward_cells_init,
												initial_states_bw=backward_cells_init,
												dtype=tf.float32,
												sequence_length=seq_length)

			with tf.variable_scope('output'):
				labels_weights = tf.get_variable('labels_weights',[2*config.acoustic_num_neurons,config.labels_dimension],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
				labels_biases = tf.get_variable('labels_biases',shape=[config.labels_dimension],dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
				
				embedding_weights = tf.get_variable('embedding_weights',[2*config.acoustic_num_neurons,2*config.embedding_size],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
				embedding_biases = tf.get_variable('embedding_biases',[2*config.embedding_size],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

				rnn_outputs = tf.reshape(rnn_outputs,[-1,2*config.acoustic_num_neurons])

				logits = tf.matmul(rnn_outputs,labels_weights) + labels_biases
				logits = tf.reshape(logits,[batch_size,-1,config.labels_dimension])

				embedding_reconstruction = tf.matmul(rnn_outputs,embedding_weights) + embedding_biases
				embedding_reconstruction = tf.reshape(embedding_reconstruction,[batch_size,-1,2*config.embedding_size])
				embedding_reconstruction = tf.nn.relu(embedding_reconstruction)

			with tf.variable_scope('classification_loss'):
				self._cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))

			with tf.variable_scope('embedding_loss'):
				self._embedding_loss = tf.reduce_mean( tf.reduce_sum(tf.squared_difference(embedding_reconstruction,phone_embedding),axis=2))

		self._loss = self._cross_entropy + config.lambda_embedding * self._embedding_loss + config.lambda_reconstrucion * self._reconstruction_loss

		if is_training:
			with tf.name_scope('optimizer'):
				#learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
				#                    config.updating_step, config.learning_decay, staircase=True)


				learning_rate = config.learning_rate
				self._learning_rate= learning_rate

				if "momentum" in config.optimizer_choice:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
				elif "adam" in config.optimizer_choice:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
				else:
					print("Optimizer must be either momentum or adam. Closing.")
					sys.exit()

				# gradient clipping
				gradients , variables = zip(*self._optimizer.compute_gradients(self._loss))
				clip_grad  = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients] 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)

		else:
			posteriors=tf.nn.softmax(logits)
			prediction=tf.argmax(logits, axis=2)
			correct = tf.equal(prediction,tf.to_int64(labels))
			accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

			self._posteriors=posteriors
			self._accuracy=accuracy
			self._prediction = prediction
		
	@property
	def loss(self):
		return self._loss

	@property
	def optimize(self):
		return self._optimize

	@property
	def cross_entropy(self):
		return self._cross_entropy

	@property
	def embedding_loss(self):
		return self._embedding_loss

	@property
	def reconstruction_loss(self):
		return self._reconstruction_loss

	@property
	def posteriors(self):
		return self._posteriors

	@property
	def correct(self):
		return self._correct

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def global_step(self):
		return self._global_step
