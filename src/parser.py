from __future__ import absolute_import, division, print_function
import nltk
import nltk.data
import os, sys
import numpy as np
import scipy.io
import json
import random
import time

def exit(message):
	print(message)
	sys.exit(-1)

def has_numbers(string):
	return any(char.isdigit() for char in string)

def get_tokens_frequencies(data, threshold=10):
	"""
		Find and return a dictionary with all the words
		having a frequency greater than threshold.

		Args:
			data (list): a list of sentences

			threshold (int): minimum frequency a word should
				have in order to be included in our dictionary

		Returns:
			filtered_tokens_freq (dict): a dictionary with 
				all the 'frequent enough' word from our data
	"""
	tokens_freq = {}

	# store the frequency of each token in tokens_freq
	for sentence in data:
		for token in sentence:
			# # ignore the token if its length is <= or it contains numbers
			if has_numbers(token):
				continue
			if token.lower() not in tokens_freq:
				tokens_freq[token.lower()] = 1
			else:
				tokens_freq[token.lower()] += 1

	# only keep the 'frequent enough' words
	filtered_tokens_freq = {}
	for token in tokens_freq:
		if tokens_freq[token] >= threshold:
			filtered_tokens_freq[token] = tokens_freq[token]

	return filtered_tokens_freq


def load_dataset_info(dataset_name, cnn_name, special_toks):
	if dataset_name not in ['flickr8k', 'flickr30k', 'mscoco']:
		exit('Dataset unknown')

	# TODO: download these from google drive if they can't be found
	dataset_path = os.path.join('..', 'data', dataset_name)
	captions_path = os.path.join(dataset_path, 'captions.json')
	features_path = os.path.join(dataset_path, 'features', 
							cnn_name + '_feats.mat')


	captions_dict = json.load(open(captions_path))
	features_dict = scipy.io.loadmat(features_path)

	num_imgs = len(captions_dict['images'])
	max_len = 30  # customize this TODO
	num_captions = 5

	all_train_sentences = []
	img2idx, idx2img = {}, {}

	for img_id in range(num_imgs):
		img_name = captions_dict['images'][img_id]['filename']
		img_split = captions_dict['images'][img_id]['split']
		img_captions = captions_dict['images'][img_id]['sentences']

		if img_split not in img2idx:
			img2idx[img_split] = {}
			idx2img[img_split] = {}

		img2idx[img_split][img_name] = img_id
		idx2img[img_split][img_id] = img_name

		if img_split == 'train':
			for sent_id in range(len(img_captions)):
				sent = img_captions[sent_id]['tokens']
				sent = [special_toks['start']] + sent + [special_toks['end']]
				all_train_sentences.append(sent) 

	all_tokens = get_tokens_frequencies(all_train_sentences)
	all_tokens[special_toks['unknown']] = 1
	voc_size = len(all_tokens)
	print('Vocabulary size %d' % voc_size)

	tok2idx = {tok : idx for idx, tok in enumerate(all_tokens)}
	idx2tok = {idx : tok for idx, tok in enumerate(all_tokens)}

	dataset_info = {}
	dataset_info['img2idx'] = img2idx
	dataset_info['idx2img'] = idx2img
	dataset_info['voc_size'] = voc_size
	dataset_info['tok2idx'] = tok2idx
	dataset_info['idx2tok'] = idx2tok
	dataset_info['all_tokens'] = all_tokens
	dataset_info['special_toks'] = special_toks
	dataset_info['max_len'] = max_len
	dataset_info['num_captions'] = num_captions
	dataset_info['all_captions_dict'] = captions_dict['images']
	dataset_info['all_features_dict'] = features_dict

	return dataset_info


def get_data_chunks(split, dataset_name, cnn_name, special_toks, dataset_info, chunk_size, show_plots=False):
	if dataset_name not in ['flickr8k', 'flickr30k', 'mscoco']:
		exit('Dataset unknown')

	if cnn_name not in ['googlenet', 'vgg-16']:
		exit('CNN name unknown')

	# length of the features vector for an image; hardcoded stuff
	feats_size = 1000 if cnn_name == 'googlenet' else 4096
	captions_per_image = 5	# take this from load_dataset_info ?
	max_len = dataset_info['max_len']
	voc_size = dataset_info['voc_size']

	replace_unknown_words = lambda s : [w if w in dataset_info['all_tokens'] \
			else special_toks['unknown'] for w in s]
	sentence_to_array = lambda s : [dataset_info['tok2idx'][w] for w in s][:max_len] + \
								   [0] * (max_len - len(s))		# padding

	# TODO: download these from google drive if they can't be found
	# dataset_path = os.path.join('..', 'data', dataset_name)
	# captions_path = os.path.join(dataset_path, 'captions.json')
	# features_path = os.path.join(dataset_path, 'features', 
	# 						cnn_name + '_feats.mat')

	# load the features dictionary and captions dictionary
	all_features_dict = dataset_info['all_features_dict']
	all_captions_dict = dataset_info['all_captions_dict']

	# all the image names in this split
	split_img_names = list(dataset_info['img2idx'][split].keys())
	num_chunks = len(split_img_names) // chunk_size

	for i in range(num_chunks):

		start2 = time.time()

		# get the names of the images that are part of the current chunk
		curr_split_img_names = split_img_names[i * chunk_size : (i + 1) * chunk_size]

		# create a dictionary img_name -> cnn_feats for all images in current chunk
		curr_split_features_dict = {img_name : feats[0] \
								for img_name, feats in all_features_dict.items() \
								if img_name in curr_split_img_names}

		# dictionary used for sorting the cnn_features (and captions later)
		img_order_dict = {curr_split_img_names[i] : i for i in range(len(curr_split_img_names))}

		# create an inverse mapping in order to be able to identify each result with the 
		# name of the image on which it is based (this is useful for testing)
		reversed_order_dict = {v : k for k, v in img_order_dict.items()}

		# create a list of tuples (img_name, cnn_feats) from the sorted dictionary
		# cnn_features[img_name] can be found on position given by img_order_dict[img_name]
		sorted_feates_list = sorted(curr_split_features_dict.items(), \
								    key = lambda i : img_order_dict.get(i[0]))

		# the previous array stored the image names too; remove them now
		sorted_feats_array = np.array([elem[1] for elem in sorted_feates_list])
		
		# get the indices of the images that are part of the current chunk
		curr_split_img_idx = [dataset_info['img2idx'][split][img_name] \
							  for img_name in curr_split_img_names]

		# sort these so that their positions match the cnn_features positions
		curr_split_img_idx = sorted(curr_split_img_idx, \
									key = lambda i : img_order_dict[dataset_info['idx2img'][split][i]])
		
		# for each image, choose a random ground truth caption
		if split != 'test':
			captions = [all_captions_dict[img_id]['sentences'][random.choice(range(5))]['tokens'] \
						for img_id in curr_split_img_idx]

			captions = [[special_toks['start']] + sent + [special_toks['end']] for sent in captions]

			# replace the unknown words (words that do not appear in the training captions) with 
			# the unknown token
			sent_filtered = [replace_unknown_words(sent) for sent in captions]
			
			# print(sent_filtered[5])

			# transform the sentence to a zero-padded array where each element is a word index 
			# (the zero-padding is at the end of the sentence and should be masked by the model)
			sent_array = np.array([sentence_to_array(sent) for sent in sent_filtered])
			
			# print(sent_array[5])

			# data_indices = caption data stored as indices
			data_indices = sent_array + 1
			data_indices[data_indices == 1] = 0

			# data one hot = caption data stored as one hot vectors; some numpy magic here
			data_one_hot = (np.arange(voc_size) == sent_array[...,None]).astype(int)

			# yield sorted_feats_array, data_indices, data_one_hot, reversed_order_dict
			yield [sorted_feats_array, data_indices[:, :-1]], data_one_hot

		else:
			captions = [all_captions_dict[img_id]['sentences'] \
						for img_id in curr_split_img_idx]
			captions = [[sent['tokens'] for sent in caps] for caps in captions]
			captions = [[' '.join(sent) for sent in caps] for caps in captions]

			yield sorted_feats_array, captions, reversed_order_dict
			

def test():
	special_toks = {}
	special_toks['start'] = '<start_tok>'
	special_toks['end'] = '<end_tok>'
	special_toks['unknown'] = '<unknown_tok>'

	dataset_info = load_dataset_info('flickr30k', special_toks)


if __name__ == '__main__':
	test()
