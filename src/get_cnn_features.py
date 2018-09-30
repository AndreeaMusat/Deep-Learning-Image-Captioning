# Andreea Musat, March 2018

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imresize, imread
from scipy.io import savemat, loadmat
from skimage.transform import resize
from os import listdir
from os.path import isfile, join

import caffe # make sure you have your caffe root in your pythonpath env var

def init_net(model='googlenet'):
	"""
	Create and return a Caffe neural network object

	Args:
		model (string): name of the model to be used [googlenet|vgg-16]

	Returns:
		net (caffe.Net): a pre-trained neural network used for extracting
			the features from the images
		layer_name (string): name of the layer from which we extract 
			the features
	"""

	caffe.set_mode_cpu()

	model_dir = os.path.join('..', 'model')
	prototxt_name = model + '.prototxt'
	prototxt_name = os.path.join(model_dir, prototxt_name)
	caffemodel_name = model + '.caffemodel'
	caffemodel_name = os.path.join(model_dir, caffemodel_name)
	
	if model == 'vgg-16':
		layer_name = 'fc7'
	elif model == 'googlenet':
		layer_name = 'loss3/classifier'
	else:
		print("Model name %s unknown. Returning..." % model)
		sys.exit(-1)

	net = caffe.Net(prototxt_name, caffemodel_name, caffe.TEST)
	return net, layer_name

def get_batch_feats(folder_name, img_list, net, layer_name):
	"""
	Run a forward pass using img_list as input and return the output
	of layer_name

	Args:
		folder_name (string): path of the directory where the images are 
			found
		img_list (list of string): a list of strings with image names 
			from folder_name
		net (caffe.Net): network to be used for feature extraction
		layer_name (string): layer from which the output is extracted

	Returns:
		output[layer_name] = numpy array of containind the features 
	"""
	N = len(img_list)
	img_mean = np.array([103.939, 116.779, 123.68])
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', img_mean)
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2, 1, 0))
	net.blobs['data'].reshape(len(img_list), 3, 224, 224)
	images = [caffe.io.load_image(folder_name+img_name) for img_name in img_list]
	images = [resize(im, (224, 224), mode='reflect') for im in images]
	transformed_imgs = [transformer.preprocess('data', img) for img in images]
	net.blobs['data'].data[0:N,:,:,:] = transformed_imgs
	output = net.forward()
	
	# for debugging
	print(output[layer_name].shape)
	print(output[layer_name])

	return output[layer_name]

def get_all_feats(folder_name, model_name, output_file, resume=True, batch_size=25):
	"""
	Compute and save the CNN features for all the images present in 'folder_name'

	A dictionary containing the name of the image mapped to its extracted 
	features will be written to output_file

	Args:
		folder_name (string): a string with the folder name in which the images 
			are found
		model_name (string): 'googlenet' | 'vgg-16'
		output_file (string): a string with the '.mat' extension in which the 
			features should be saved
		resume (bool): should be True if we've already computed the features for 
			some of the images
		batch_size (int): size of batch to be fed into the neural net
	"""

	all_features = loadmat(output_file) if resume and os.path.exists(output_file) else {}
	img_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
	img_files = [f for f in img_files if f not in all_features] 
	net, layer_name = init_net(model=model_name)
	if len(img_files) == 0: return
	for i in range(1 + int(len(img_files)/batch_size)):
		curr_batch = img_files[i*batch_size:min((i+1)*batch_size, len(img_files))]
		curr_feats = get_batch_feats(folder_name, curr_batch, net, layer_name)
		all_features.update(dict(zip(curr_batch, curr_feats)))
		savemat(output_file, mdict=all_features)


if __name__ == '__main__':
	
	if len(sys.argv) != 4:
		print("Usage: python get_features.py folder_name [googlenet|vgg-16] output_file")
		sys.exit(-1)

	folder_name, model_name, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
	get_all_feats(folder_name, model_name, output_file)
