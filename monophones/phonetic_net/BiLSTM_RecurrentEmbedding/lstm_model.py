from __future__ import division
import tensorflow as tf
import sys

class Model(object):

	def __init__(self,phone_features,acoustic_features,seq_length,config,is_training):

		if is_training:
			batch_size=config.batch_size
			dropout_keep_prob=0.8
		else:
			batch_size=1
			dropout_keep_prob=1.0

		global_step = tf.Variable(0, trainable=False)
		self._global_step=global_step

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

				embedding = tf.concat(embedding_output,2,name='recurrent_embedding')

		# lstm cells definition
		with tf.variable_scope('forward'):

			forward_cells = []
			forward_cells_init = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_forward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.n_hidden,
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
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_backward =tf.contrib.rnn.LayerNormBasicLSTMCell(config.n_hidden,
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
											inputs=embedding,
											initial_states_fw=forward_cells_init,
											initial_states_bw=backward_cells_init,
											dtype=tf.float32,
											sequence_length=seq_length)


		with tf.variable_scope('output'):
			output_weights = tf.get_variable('output_weights',[2*config.n_hidden,config.acoustic_feat_dimension],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases = tf.get_variable('biases',shape=[config.acoustic_feat_dimension],dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			rnn_outputs = tf.reshape(rnn_outputs,[-1,2*config.n_hidden])	
			
			acoustic_reconstruction = tf.matmul(rnn_outputs,output_weights) + output_biases
		
			acoustic_reconstruction = tf.reshape(acoustic_reconstruction,[batch_size,-1,config.acoustic_feat_dimension])

		with tf.name_scope('loss'):
			diff = tf.squared_difference(acoustic_reconstruction,acoustic_features)
			self._loss = tf.reduce_mean( tf.reduce_sum(diff,axis=2) )

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

		
	@property
	def loss(self):
		return self._loss

	@property
	def optimize(self):
		return self._optimize


	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def global_step(self):
		return self._global_step
