# Avoid printing tensorflow log messages
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import time
import sys

import lstm_model 

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])

num_examples=3696

class Configuration(object):
	
	learning_rate=float(sys.argv[2])
	batch_size=int(sys.argv[3])
	optimizer_choice=sys.argv[4]

	acoustic_feat_dimension = 120
	phonetic_feat_dimension = 43
	labels_dimension=144
	embedding_size=75

	num_epochs=5000
	
	phonetic_num_neurons=100
	phonetic_num_layers=2

	acoustic_num_neurons=250
	acoustic_num_layers=5

	lambda_embedding=float(sys.argv[5])
	lambda_reconstrucion=float(sys.argv[6])

	num_examples_val=400

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

tensorboard_dir='tensorboard/exp'+str(ExpNum)+'/'

trainingLogFile=open('TrainingExperiment'+str(ExpNum)+'.txt','w')

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


###################################
# Auxiliary functions
###################################

# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,config):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"acoustic_feat":tf.FixedLenSequenceFeature([config.acoustic_feat_dimension],dtype=tf.float32),
                                                                  "labels":tf.FixedLenSequenceFeature([],dtype=tf.float32),
                                                                  "phonetic_feat":tf.FixedLenSequenceFeature([config.phonetic_feat_dimension],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['acoustic_feat'],sequence_parsed['phonetic_feat'],tf.to_int32(sequence_parsed['labels'])

# training input pipeline
def input_pipeline(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=config.num_epochs, shuffle=True)
    
    sequence_length, acoustic_features, phonetic_features, audio_labels = read_my_file_format(filename_queue,config)

    acoustic_features_batch, audio_labels_batch ,\
    phonetic_features_batch, seq_length_batch = tf.train.batch([acoustic_features, audio_labels, phonetic_features, sequence_length],
                                                    batch_size=config.batch_size,
                                                    num_threads=10,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return acoustic_features_batch, phonetic_features_batch, audio_labels_batch, seq_length_batch

def input_pipeline_validation(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)
    
    sequence_length, acoustic_features, phonetic_features, audio_labels = read_my_file_format(filename_queue,config)

    acoustic_features_batch, audio_labels_batch ,\
    phonetic_features_batch, seq_length_batch = tf.train.batch([acoustic_features, audio_labels, phonetic_features, sequence_length],
                                                    batch_size=1,
                                                    num_threads=10,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return acoustic_features_batch, phonetic_features_batch, audio_labels_batch, seq_length_batch

#################################
# Training module
#################################
def train():

	config=Configuration()

	# list of input filenames + check existence
	filename_train=['/home/local/IIT/rtavarone/PhoneticContextEmbedding/data_processing/TRAIN/sequence_{:04d}.tfrecords'.format(i) \
					 for i in range(num_examples)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	filename_val=['/home/local/IIT/rtavarone/PhoneticContextEmbedding/data_processing/VAL/sequence_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_examples_val)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)


	# training graph		
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			with tf.name_scope('train_batch'):
				acoustic_features, phonetic_features, audio_labels, seq_length = input_pipeline(filename_train,config)

		
		with tf.device('/cpu:0'):
			with tf.name_scope('validation_batch'):
				acoustic_features_val, phonetic_features_val, audio_labels_val, seq_length_val = input_pipeline_validation(filename_val,config)
		
		# audio features reconstruction
		with tf.device('/cpu:0'):
			with tf.variable_scope('model',reuse=None):
				print('Building training model:')
				train_model = lstm_model.Model(phonetic_features,acoustic_features,audio_labels,seq_length,config,is_training=True)
				print('done.\n')

		
		with tf.device('/cpu:0'):
			with tf.variable_scope('model',reuse=True):
				print('Building validation model:')
				val_model = lstm_model.Model(phonetic_features_val,acoustic_features_val,audio_labels_val,seq_length_val,config,is_training=None)
				print('done.')
		

		# variables initializer
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=10)

		
		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			# tensorboard writer
			#train_writer = tf.summary.FileWriter(tensorboard_dir,sess.graph)

			# run initializer
			sess.run(init_op)


			# start queue coordinator and runners
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)


			#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			#sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

			print('')
			print('## EXPERIMENT NUMBER ',ExpNum)
			trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))
			
			print('## optimizer : ',config.optimizer_choice)
			trainingLogFile.write('## optimizer : {:s} \n'.format(config.optimizer_choice))
			
			print('## number of hidden layers phonetic net: ',config.phonetic_num_layers)
			trainingLogFile.write('## number of hidden layers phonetic net: {:d} \n'.format(config.phonetic_num_layers))
			
			print('## number of hidden units phonetic net: ',config.phonetic_num_neurons)
			trainingLogFile.write('## number of hidden units phonetic net: {:d} \n'.format(config.phonetic_num_neurons))

			print('## number of hidden layers acoustic net: ',config.acoustic_num_layers)
			trainingLogFile.write('## number of hidden layers acoustic net: {:d} \n'.format(config.acoustic_num_layers))
			
			print('## number of hidden units acoustic net: ',config.acoustic_num_neurons)
			trainingLogFile.write('## number of hidden units acoustic net: {:d} \n'.format(config.acoustic_num_neurons))
			
			print('## phonetic embedding size : ',config.embedding_size)
			trainingLogFile.write('## phonetic embedding size: {:d} \n'.format(config.embedding_size))

			print('## lambda reconstruction : ',config.lambda_reconstrucion)
			trainingLogFile.write('## lambda reconstruction: {:f} \n'.format(config.lambda_reconstrucion))

			print('## lambda embedding : ',config.lambda_embedding)
			trainingLogFile.write('## lambda embedding: {:f} \n'.format(config.lambda_embedding))			

			print('## learning rate : ',config.learning_rate)
			trainingLogFile.write('## learning rate : {:.6f} \n'.format(config.learning_rate))
			
			print('## batch size : ',config.batch_size)
			trainingLogFile.write('## batch size : {:d} \n'.format(config.batch_size))
			
			print('## number of steps: ',num_examples*config.num_epochs/config.batch_size)
			trainingLogFile.write('## approx number of steps: {:d} \n'.format(int(num_examples*config.num_epochs/config.batch_size)))
			
			print('## number of steps per epoch: ',num_examples/config.batch_size)
			trainingLogFile.write('## approx number of steps per epoch: {:d} \n'.format(int(num_examples/config.batch_size)))
			print('')

			try:
				epoch_counter=1
				epoch_loss=0.0
				epoch_ce = 0.0
				epoch_embedding = 0.0
				epoch_reconstrucion = 0.0

				epoch_steps=0
				EpochStartTime=time.time()
				partial_time=time.time()

				step=1

				while not coord.should_stop():

					_ , loss, ce_loss, embedding_loss , reconstruction_loss  = sess.run([train_model.optimize,train_model.loss,
																					train_model.cross_entropy,
																					train_model.embedding_loss,
																					train_model.reconstruction_loss])


					epoch_loss += loss
					epoch_ce += ce_loss
					epoch_embedding += embedding_loss
					epoch_reconstrucion += reconstruction_loss

					epoch_steps += 1
					
					if (step % 100 == 0 or step==1):
						print("step[{:7d}] loss[{:2.5f}] cross_entropy[{:2.5f}] embedding_loss[{:2.5f}] reconstruction_loss[{:2.5f}] time[{}]".\
								format(step,loss,ce_loss,embedding_loss,reconstruction_loss,time.time()-partial_time))
						partial_time=time.time()

					'''
					if (step%500==0):
						# save training parameters
						save_path = saver.save(sess,checkpoints_dir+'model_step'+str(step)+'.ckpt')
						print('Model saved!')
					'''

					if ((step % int(num_examples / config.batch_size) == 0) and (step is not 0)):

						# at each step we get the average cost over a batch, so divide by the 
						# number of batches in one epoch
						epoch_loss /=  epoch_steps
						epoch_ce /= epoch_steps
						epoch_embedding /= epoch_steps
						epoch_reconstrucion /= epoch_steps

						print('')
						print('Completed epoch {:d} at step {:d}'.format(epoch_counter,epoch_steps))
						print('Epoch training time (seconds) = ',time.time()-EpochStartTime)
						print('Training:')
						print('loss[{:2.5f}] cross_entropy[{:2.5f}] embedding_loss[{:2.5f}] reconstruction_loss[{:2.5f}]'.format(epoch_loss,
																																epoch_ce,
																																epoch_embedding,
																																epoch_reconstrucion))
						
						
						#accuracy evaluation on each sentence
						#to avoid computing accuracy on padded frames
						
						out_every_epoch=1
							
						if((epoch_counter%out_every_epoch)==0):

							validation_time=time.time()
							val_loss = 0.0
							val_ce = 0.0
							val_embedding = 0.0
							val_reconstrucion = 0.0
							val_accuracy=0.0

							for i in range(config.num_examples_val):

								# validation
								loss, ce_loss, embedding_loss , reconstruction_loss , example_accuracy = sess.run([val_model.loss,
																					val_model.cross_entropy,
																					val_model.embedding_loss,
																					val_model.reconstruction_loss,
																					val_model.accuracy])

								
								val_loss += loss
								val_ce += ce_loss
								val_embedding += embedding_loss
								val_reconstrucion += reconstruction_loss
								val_accuracy += example_accuracy


							val_loss /= config.num_examples_val
							val_ce /= config.num_examples_val
							val_embedding /= config.num_examples_val
							val_reconstrucion /= config.num_examples_val
							val_accuracy /= config.num_examples_val
							
							# printout validation results
							print('')
							print('Validation: ')
							print('loss[{:2.5f}] cross_entropy[{:2.5f}] embedding_loss[{:2.5f}] reconstruction_loss[{:2.5f}] accuracy[{:2.5}] time[{}]'.format(val_loss,
																																val_ce,
																																val_embedding,
																																val_reconstrucion,
																																val_accuracy,
																																time.time()-validation_time))

							
							trainingLogFile.write('{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(epoch_counter,epoch_loss,val_loss,val_ce,val_embedding,val_reconstrucion,val_accuracy))
							trainingLogFile.flush()

							save_path = saver.save(sess,checkpoints_dir+'model_epoch'+str(epoch_counter)+'.ckpt')
						
						print('\n')	
						epoch_counter+=1
						epoch_steps=0

						epoch_loss=0.0
						epoch_ce =0.0
						epoch_embedding = 0.0
						epoch_reconstrucion = 0.0
						

						EpochStartTime=time.time()

					step += 1

			except tf.errors.OutOfRangeError:
				print('---- Done Training: epoch limit reached ----')
			finally:
				coord.request_stop()

			coord.join(threads)

			save_path = saver.save(sess,checkpoints_dir+'model_end.ckpt')
			print("model saved in file: %s" % save_path)

	trainingLogFile.close()


def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()