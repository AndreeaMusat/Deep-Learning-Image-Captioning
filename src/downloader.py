# Andreea Musat, April 2018

import argparse
import sys, os
curr_dir_name = '.'
downloader_dir_name = 'google_drive_downloader'

sys.path.append(os.path.join(
	curr_dir_name, 
	downloader_dir_name
))

from google_drive_downloader import GoogleDriveDownloader as gdd
from utils import exit


def download_model(model_name):
	"""
		TODO: add documentation
	"""

	# info about the files to be downloaded	
	caffemodel_file_name = model_name + '.caffemodel'
	prototxt_file_name = model_name + '.prototxt'

	if model_name == 'googlenet':
		file_id = '1_ESR5hoUfStkKj0Srj7lFT7JnL2-OxKs'
	elif model_name == 'vgg-16':
		file_id = '1UTHL_cyPGl4ZhDMmaVZ5w0vuqfGUTh_p'
	else:
		message = '[ERROR] Model file ' + model_name + ' unknown'
		exit(message, -1)
	
	caffemodel_path = os.path.join('..', 'model', caffemodel_file_name)
	prototxt_path = os.path.join('..', 'model', prototxt_file_name)

	if os.path.isfile(caffemodel_path) and \
	   os.path.isfile(prototxt_path):
		message = '[SUCCESS] Model files already downloaded. Exiting.'
		exit(message, 0)

	# download the zip file with the prototxt and caffemodel and unzip
	model_path = os.path.join('..', 'model', 'junk.zip')
	gdd.download_file_from_google_drive(
		file_id, 
		dest_path=model_path,
		unzip=True
	)

	# now remove the zip file as it's not necessary anymore
	os.remove(model_path)

	message = '[SUCCESS] Model ' + model_name + ' has been downloaded.'
	exit(message, 0)


def donwload_cnn_features(cnn_name, dataset_name):
	"""
		TODO: add doc
	"""

	if dataset_name == 'flickr8k' and cnn_name == 'googlenet':
		file_id = '1UMSdcs8j_nooICgerF_KhLC_0aoa6Q_-'
	elif dataset_name == 'flickr8k' and cnn_name == 'vgg-16':
		file_id = '12I4HfxBdzSdgYFJe0SwijEwe91tHz-9D'
	elif dataset_name == 'flickr30k' and cnn_name == 'googlenet':
		file_id = '1onveJD-bW2gInn0tBUM2-SXnqfcqNak3'
	elif dataset_name == 'flickr30k' and cnn_name == 'vgg-16':
		file_id = '15Dc6KYev9ogYmLSXw98A3ZrE2zou6Bmu'
	else:
		# TODO: add the other datasets
		exit('Not yet implemented.', -1)

	file_name = cnn_name + '_feats' + '.mat'
	file_path = os.path.join('..', 'data', dataset_name, 'features', file_name)

	if os.path.isfile(file_path):
		message = '[SUCCESS] Cnn features already downloaded. Exiting'
		exit(message, 0)

	gdd.download_file_from_google_drive(
		file_id, 
		dest_path=file_path, 
		unzip=False
	)

	message = '[SUCCES] ' + cnn_name + ' features for dataset ' + dataset_name +\
			  ' have been downloaded. Exiting'
	exit(message, 0)


def download_captions(dataset_name):
	"""
		TODO: add doc
	"""

	if dataset_name == 'flickr8k':
		file_id = '1QQ2WukR9SyMhhIHkgNQqy4QORUr71rWJ'
	elif dataset_name == 'flickr30k':
		file_id = '1sP_-PRIfUULV6SnJkyrlzcjycawG0gss'
	else:
		# TODO: add the other datasets
		exit('[ERROR] Not yet implemented', -1)

	file_name = 'captions.json'
	file_path = os.path.join('..', 'data', dataset_name, file_name)

	if os.path.isfile(file_path):
		message = '[SUCCESS] ' + file_path + ' already exists. Exiting'
		exit(message, 0)

	gdd.download_file_from_google_drive(
		file_id, 
		dest_path=file_path,
		unzip=False
	)

	message = '[SUCCESS] Captions for dataset ' + dataset_name +\
	 ' have been downloaded. Exiting'
	exit(message, 0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--download', type=str, dest='download', 
			help='cnn|cnn_features|captions')
	parser.add_argument('--dataset-name', type=str, dest='dataset_name', 
			help='flickr8k|flickr30k|mscoco')
	parser.add_argument('--cnn-name', type=str, dest='cnn_name', 
			help='googlenet|vgg-16')

	args, unknown = parser.parse_known_args()

	if args.download == 'cnn':
		if args.cnn_name is not None:
			download_model(args.cnn_name)
		else:
			exit('[ERROR] cnn_model should be googlenet|vgg-16', -1)
	elif args.download == 'cnn_features':
		if args.cnn_name is not None and args.dataset_name is not None:
			donwload_cnn_features(args.cnn_name, args.dataset_name)
		elif args.cnn_name is None:
			exit('[ERROR] cnn_model should be googlenet|vgg-16', -1)
		elif args.dataset_name is None:
			exit('[ERROR] dataset_name should be flickr8k|flickr30k|mscoco', -1)
	elif args.download == 'captions':
		if args.dataset_name is not None:
			download_captions(args.dataset_name)
		else:
			exit('[ERROR] dataset_name should be flickr8k|flickr30k|mscoco', -1)
