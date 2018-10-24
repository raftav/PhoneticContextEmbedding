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
restore_epoch = int(sys.argv[2])

class Configuration(object):
	
	acoustic_feat_dimension = 120
	phonetic_feat_dimension = 43

	labels_dimension=1944

	acoustic_num_neurons=250
	acoustic_num_layers=5

	num_examples_test=192

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

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

def input_pipeline_validation(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    
    sequence_length, acoustic_features, phonetic_features, audio_labels = read_my_file_format(filename_queue,config)

    acoustic_features_batch, audio_labels_batch ,\
    phonetic_features_batch, seq_length_batch = tf.train.batch([acoustic_features, audio_labels, phonetic_features, sequence_length],
                                                    batch_size=1,
                                                    num_threads=10,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return acoustic_features_batch, phonetic_features_batch, audio_labels_batch, seq_length_batch


def test():

	config=Configuration()

	filename_test=['/home/local/IIT/rtavarone/PhoneticContextEmbedding/triphones/data/TEST/sequence_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_examples_test)]
	for f in filename_test:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			with tf.name_scope('validation_batch'):
				acoustic_features_test, phonetic_features_test, audio_labels_test, seq_length_test = input_pipeline_validation(filename_test,config)
		
		with tf.device('/cpu:0'):
			with tf.variable_scope('model'):
				print('Building validation model:')
				test_model = lstm_model.Model(acoustic_features_test,audio_labels_test,seq_length_test,config,is_training=False)
				print('done.')

		# variables initializer
		init_op = tf.local_variables_initializer()

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=None)

		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			sess.run(init_op)
			print('Restoring variables...')
			saver.restore(sess,checkpoints_dir+'model_epoch'+str(restore_epoch)+'.ckpt')
			print('Model loaded')

			# start queue coordinator and runners
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			try:
				count=0
				accuracy=0.0
				while not coord.should_stop():

					example_accuracy  = sess.run(test_model.accuracy)
					accuracy += example_accuracy
					print('sentence [{}] accuracy[{}]'.format(count,example_accuracy))
					count+=1

			except tf.errors.OutOfRangeError:
				accuracy /= count
				print('Total number sentence = ',count)
				print('Test accuracy = ',accuracy)

			finally:
				coord.request_stop()

def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  tf.app.run()