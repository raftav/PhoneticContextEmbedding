import numpy as np
from itertools import groupby
import tensorflow as tf
import math
import pickle
import glob
import sys

def serialize_sequence(audio_sequence,phonetic_sequence,labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(audio_sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)

    # Feature lists for the two sequential features of our example
    fl_acoustic_feat = ex.feature_lists.feature_list["acoustic_feat"]
    fl_phonetic_feat = ex.feature_lists.feature_list["phonetic_feat"]
    fl_labels = ex.feature_lists.feature_list["labels"]

    for acoustic_feat, phonetic_feat, label in zip(audio_sequence, phonetic_sequence,labels):
        fl_acoustic_feat.feature.add().float_list.value.extend(acoustic_feat)
        fl_phonetic_feat.feature.add().float_list.value.extend(phonetic_feat)
        fl_labels.feature.add().float_list.value.append(label)
    return ex

##############################
##############################

training_files = glob.glob('/home/local/IIT/lbadino/Data/TFP/timit_pce/rnn/data/train/*.npy')
train_files_list = open('training_files_list.txt','w')

label_max=0

for index,file in enumerate(training_files):

	train_files_list.write(file+'\t'+str(index)+'\n')

	data = np.load(file)
	#data = data.astype(np.float)
	print(index,file)
	filename='TRAIN/sequence_{:04d}.tfrecords'.format(index)

	fp = open(filename,'w')
	writer = tf.python_io.TFRecordWriter(fp.name)

	acoustic_feat=data[:,0:120]
	print('acoustic feat shape = ',acoustic_feat.shape)

	phonetic_feat=data[:,121:]
	print('phonetic_feat shape = ',phonetic_feat.shape)

	label_file=file.replace('/home/local/IIT/lbadino/Data/TFP/timit_pce/rnn/data/train/','/home/local/IIT/lbadino/Data/KP/timit/data/train/')
	tri_labels=np.load(label_file)
	tri_labels=tri_labels[:,120]
	temp_label_max=np.amax(tri_labels)
	if temp_label_max > label_max:
		label_max=temp_label_max

	if tri_labels.shape[0] == acoustic_feat.shape[0]:
		serialized_sentence = serialize_sequence(acoustic_feat,phonetic_feat,tri_labels)
	else:
		print('unmatched sentence lenght. exit.')
		sys.exit()

	# write to tfrecord
	writer.write(serialized_sentence.SerializeToString())
	writer.close()

	fp.close()

train_files_list.close()

print('Training data processing done')
print('max value triphone labels = ',label_max)

##############################
##############################

validation_files = glob.glob('/home/local/IIT/lbadino/Data/TFP/timit_pce/rnn/data/dev/*.npy')
validation_files_list = open('validation_files_list.txt','w')

for index,file in enumerate(validation_files):

	validation_files_list.write(file+'\t'+str(index)+'\n')

	data = np.load(file)
	#data = data.astype(np.float)
	print(index,file)
	filename='VAL/sequence_{:04d}.tfrecords'.format(index)

	fp = open(filename,'w')
	writer = tf.python_io.TFRecordWriter(fp.name)

	acoustic_feat=data[:,0:120]
	print('acoustic feat shape = ',acoustic_feat.shape)

	phonetic_feat=data[:,121:]
	print('phonetic_feat shape = ',phonetic_feat.shape)

	label_file=file.replace('/home/local/IIT/lbadino/Data/TFP/timit_pce/rnn/data/dev/','/home/local/IIT/lbadino/Data/KP/timit/data/dev/')
	tri_labels=np.load(label_file)
	tri_labels=tri_labels[:,120]

	if tri_labels.shape[0] == acoustic_feat.shape[0]:
		serialized_sentence = serialize_sequence(acoustic_feat,phonetic_feat,tri_labels)
	else:
		print('unmatched sentence lenght. exit.')
		sys.exit()

	# write to tfrecord
	writer.write(serialized_sentence.SerializeToString())
	writer.close()

	fp.close()

validation_files_list.close()

print('Validation data processing done')
##############################
##############################

test_files = glob.glob('/home/local/IIT/lbadino/Data/TFP/timit_pce/rnn/data/test/*.npy')
test_files_list = open('test_files_list.txt','w')

for index,file in enumerate(test_files):

	test_files_list.write(file+'\t'+str(index)+'\n')

	data = np.load(file)
	#data = data.astype(np.float)
	print(index,file)
	filename='TEST/sequence_{:04d}.tfrecords'.format(index)

	fp = open(filename,'w')
	writer = tf.python_io.TFRecordWriter(fp.name)

	acoustic_feat=data[:,0:120]
	print('acoustic feat shape = ',acoustic_feat.shape)

	phonetic_feat=data[:,121:]
	print('phonetic_feat shape = ',phonetic_feat.shape)

	label_file=file.replace('/home/local/IIT/lbadino/Data/TFP/timit_pce/rnn/data/test/','/home/local/IIT/lbadino/Data/KP/timit/data/test/')
	tri_labels=np.load(label_file)
	tri_labels=tri_labels[:,120]
	
	if tri_labels.shape[0] == acoustic_feat.shape[0]:
		serialized_sentence = serialize_sequence(acoustic_feat,phonetic_feat,tri_labels)
	else:
		print('unmatched sentence lenght. exit.')
		sys.exit()


	# write to tfrecord
	writer.write(serialized_sentence.SerializeToString())
	writer.close()

	fp.close()

test_files_list.close()