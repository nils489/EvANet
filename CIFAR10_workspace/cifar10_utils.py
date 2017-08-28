#!/usr/bin/env python

import os
import urllib.request
import numpy as np
import tensorflow as tf
import progressbar
import tarfile
import pickle

bar = progressbar.ProgressBar(max_value=100)

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo,fix_imports=False,encoding='Latin1')
	return dict

def dlProgress(count, blockSize, totalSize):
	percent = int(count*blockSize*100/totalSize)
	bar.update(percent)
	
def read_cifar10_tensorflow():
	data_batches = []
	label_batches = []
	tmp_dict = {}
	tmp_dict = unpickle("cifar-10-batches-py/data_batch_1")	
	data_batches.append(tmp_dict['data'])
	label_batches.append(tmp_dict['labels'])
	tmp_dict = unpickle("cifar-10-batches-py/data_batch_2")	
	data_batches.append(tmp_dict['data'])
	label_batches.append(tmp_dict['labels'])
	tmp_dict = unpickle("cifar-10-batches-py/data_batch_3")	
	data_batches.append(tmp_dict['data'])
	label_batches.append(tmp_dict['labels'])
	tmp_dict = unpickle("cifar-10-batches-py/data_batch_4")	
	data_batches.append(tmp_dict['data'])
	label_batches.append(tmp_dict['labels'])
	tmp_dict = unpickle("cifar-10-batches-py/data_batch_5")	
	data_batches.append(tmp_dict['data'])
	label_batches.append(tmp_dict['labels'])
	tmp_dict = unpickle("cifar-10-batches-py/test_batch")	
	data_batches.append(tmp_dict['data'])
	label_batches.append(tmp_dict['labels'])
	X = np.concatenate((data_batches[0],data_batches[1],data_batches[2],data_batches[3],data_batches[4]))
	y = np.concatenate((np.array(label_batches[0]),np.array(label_batches[1]),np.array(label_batches[2]),np.array(label_batches[3]),np.array(label_batches[4])))
	y = np.array(y,dtype=int)
	x_test = data_batches[5]
	y_test = label_batches[5]
	return({'data':X,'labels':y,'data_test':x_test,'labels_test':y_test})
	

def get_cifar10():
	CIFAR10_FILENAME = "cifar-10-python.tar.gz"
	CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	
	if not os.path.exists(CIFAR10_FILENAME):
		print("Downloading CIFAR-10...")
		urllib.request.urlretrieve(CIFAR10_URL,CIFAR10_FILENAME, reporthook=dlProgress)

	# opening .tar.gz file
	archive = tarfile.open(CIFAR10_FILENAME,'r:gz')
	archive.extractall()
	return(read_cifar10_tensorflow())
