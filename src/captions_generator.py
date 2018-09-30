from __future__ import print_function

import numpy as np
import time
import heapq
import keras
import pickle 

# from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
from scipy.io import savemat

from evaluate import eval_captions
from models import get_model
from models import get_embed_matrix
from utils import FixedCapacityMaxHeap, exit
from new_parser import get_data_chunks

class CaptionsGenerator(object):

	def __init__(self, args, dataset_info):
		self.dataset_name = args.dataset_name
		self.model_name = args.model_name
		self.cnn_name = args.cnn_name
		self.embed_size = args.embed_size
		self.batch_size = args.batch_size
		self.embed_type = 'default'
		self.dropout = 0.3
		self.learning_rate = args.learning_rate
		self.eval_every = args.eval_every
		self.num_iterations = args.num_iterations
		self.output_name = args.plot_name
		self.use_beam_search = args.use_beam_search

		self.tok2idx = dataset_info['tok2idx']
		self.idx2tok = dataset_info['idx2tok']
		self.img2idx = dataset_info['img2idx']
		self.idx2img = dataset_info['idx2img']
		self.max_len = dataset_info['max_len']
		self.voc_size = dataset_info['voc_size']
		self.special_toks = dataset_info['special_toks']
		self.num_captions = dataset_info['num_captions']
		self.dataset_info = dataset_info

		self.end_idx = self.tok2idx[self.special_toks['end']]
		self.start_idx = self.tok2idx[self.special_toks['start']]
		self.unknown_idx = self.tok2idx[self.special_toks['unknown']]

		if args.cnn_name == 'googlenet':
			self.cnn_feats_size = 1000
		elif args.cnn_name == 'vgg-16':
			self.cnn_feats_size = 4096

		embed_matrix = get_embed_matrix(
		   self.voc_size, 
		   self.embed_size, 
		   self.embed_type,
		   self.tok2idx,
		   self.idx2tok
		)

		self.model = get_model(
			self.model_name, 
			self.batch_size,
			self.learning_rate, 
			self.max_len - 1, 
			self.voc_size, 
			self.embed_size, 
			self.cnn_feats_size, 
			self.dropout, 
			embed_matrix,
		)

		self.train_losses = []
		self.train_acc = []
		self.val_losses = []
		self.val_acc = []
		self.eval_scores = {}

	def plot_loss(self, history):
		self.train_losses += history.history['loss']
		self.train_acc += history.history['acc']
		self.val_losses += history.history['val_loss']
		self.val_acc += history.history['val_acc']

		plot_name = self.output_name + '_loss.png'
		plt.plot(self.train_losses)
		plt.plot(self.val_losses)
		plt.xlabel('Loss')
		plt.ylabel('Epoch')
		plt.legend(['train', 'validation'], loc='best')
		plt.savefig(plot_name)
		plt.close()

		plot_name = self.output_name + '_accuracy.png'
		plt.plot(self.train_acc)
		plt.plot(self.val_acc)
		plt.xlabel('Accuracy')
		plt.ylabel('Epoch')
		plt.legend(['train', 'validation'], loc='best')
		plt.savefig(plot_name)
		plt.close()

	def remove_zero_pad(self, arr):
		zero_pad_pos = np.argwhere(arr == 0)

		if len(zero_pad_pos) == 0:
			return arr

		first_zero_pad_idx = zero_pad_pos[0][0]
		arr = arr[:first_zero_pad_idx]

		return list(arr)

				
	def beam_search(self, cnn_feats, hypotheses, t):
		chunk_size = len(cnn_feats)

		if t == self.max_len - 2:
			for i in range(chunk_size):
				for prob, caption in hypotheses[i].get_best():
					self.best_captions[i].push(list(caption), prob)
			return
		else:
			# For every candidate caption of length t (there should be self.max_heap_size candidates)
			# augment the caption with a new word.
			new_hypotheses = [FixedCapacityMaxHeap(self.max_heap_size) for i in range(chunk_size)]
			
			print(hypotheses[0].get_best()[0])

			for i in range(len(hypotheses[0].h)):	# curr path index
				test_x = np.zeros((chunk_size, self.max_len - 1), dtype=np.int)
				probs = np.zeros(chunk_size, dtype=np.float)

				for j in range(chunk_size):
					i_th_best_prob, i_th_best_caption = hypotheses[j].get_best()[i]
					test_x[j, :] = np.array(i_th_best_caption)
					probs[j] = i_th_best_prob

				# Make predictions.
				pred = self.model.predict(x=[cnn_feats, test_x], 
										  batch_size=256, 
										  verbose=1)[:, t + 1 , :]
				
				# Get the top indices (token idx with highest prob) & their probabilities.
				top_idx = np.argpartition(-pred, range(self.beam_size), axis=1)[:, :self.beam_size]
				top_probs = pred[np.arange(top_idx.shape[0])[:, None], top_idx]

				for j in range(self.beam_size):		# candidate idx
					curr_idx = top_idx[:, j]
					curr_probs = top_probs[:, j]

					new_caption = np.copy(test_x)
					new_caption[:, t + 1] = curr_idx + 1

					new_probs = probs * curr_probs

					for k in range(chunk_size):
						new_hypotheses[k].push(list(new_caption[k]), new_probs[k])

			self.beam_search(cnn_feats, new_hypotheses, t + 1)

	def generate_captions(self):
		
		chunk_size = 1000
		test_data_generator = get_data_chunks(
			split='test',
			dataset_name=self.dataset_name,
			cnn_name=self.cnn_name,
			special_toks=self.special_toks,
			dataset_info=self.dataset_info,
			chunk_size=1000,
		)

		self.max_heap_size = 20
		self.beam_size = 3

		argmax_captions = {}
		beam_3_captions = {}
		ground_truth_captions = {}

		for cnn_feats, captions, name_dict in test_data_generator:
			# Beam search.
			if self.use_beam_search:
				hypotheses = [FixedCapacityMaxHeap(self.max_heap_size) for i in range(chunk_size)]
				print('We have ', len(hypotheses), 'hypo')

				# Start caption should only contain the start token.
				start_caption = np.zeros((1, self.max_len - 1), dtype=np.int)
				start_caption[0, 0] = self.start_idx + 1

				for i in range(chunk_size):
					hypotheses[i].push(list(np.copy(start_caption)), 1)

				print('HYPO BEFORE BEAM ', hypotheses[0].h[0])

				np.set_printoptions(threshold=np.nan)

				self.best_captions = [FixedCapacityMaxHeap(self.max_heap_size) for i in range(chunk_size)]
				self.beam_search(cnn_feats, hypotheses, 0)
				
				for img_idx, img_name in name_dict.items():
					prob, caption = self.best_captions[img_idx].get_best()[0]
					caption = list(self.remove_zero_pad(np.array(caption)))  
					caption = list(filter(lambda x : x != self.unknown_idx - 1, caption))
					caption = [self.idx2tok[idx - 1] for idx in caption[1:]] # captions are +1; remove start tok 

					if self.idx2tok[self.end_idx] in caption:
						caption = caption[:caption.index(self.idx2tok[self.end_idx])]

					caption = list(filter(lambda x : x != self.idx2tok[self.unknown_idx], caption))
					caption = [' '.join(caption)]

					beam_3_captions[img_name] = caption
					ground_truth_captions[img_name] = captions[img_idx]

			# Argmax search.
			else:
				test_x = np.zeros((chunk_size, self.max_len - 1), dtype=np.int)
				test_x[:, 0] = self.start_idx + 1

				for t in range(self.max_len - 1):
					pred = self.model.predict(
						x=[cnn_feats, test_x],
						verbose=1,
						batch_size=128,
					)[:, t + 1, :]

					if t + 1 != self.max_len - 1:
						for k in range(chunk_size):
							gen_idx = np.argmax(pred[k])
							test_x[k, t + 1] = gen_idx + 1

				for img_idx, img_name in name_dict.items():
					# Remove zero pad and subtract one from each idx (as the indices fed
					# into the LSTM are the real tok idx + 1).
					my_caption = list(self.remove_zero_pad(test_x[img_idx, 1:]) - 1)
			
					# Remove end token from the generated caption if present.
					if self.end_idx in my_caption:
						my_caption = my_caption[:my_caption.index(self.end_idx)]

					# Remove unknown tokens from my caption and ground truth caption.
					my_caption = list(filter(lambda x : x != self.unknown_idx, my_caption))

					# Go from idx to sentences.
					my_caption = [self.idx2tok[idx] for idx in my_caption]
					my_caption = [' '.join(my_caption)]
					argmax_captions[img_name] = my_caption
					ground_truth_captions[img_name] = captions[img_idx]

		# Save the captions in order to be able to visualize them later.
		captions_file_name = self.output_name + '_captions.mat'
		savemat(captions_file_name, argmax_captions)

		# Return ground truth & generated captions.
		if self.use_beam_search:
			return ground_truth_captions, argmax_captions
		else:
			return ground_truth_captions, beam_3_captions


	def evaluate(self):
		ground_truth_captions, generated_captions = self.generate_captions()
		evaluations = eval_captions(ground_truth_captions, generated_captions)

		if len(self.eval_scores) == 0:
			for i in range(len(evaluations)):
				self.eval_scores[evaluations[i][0]] = [evaluations[i][1]]
		else:
			for i in range(len(evaluations)):
				self.eval_scores[evaluations[i][0]].append(evaluations[i][1])

		color = {'Bleu_1' : 'blue', 'Bleu_2' : 'cornflowerblue', 'Bleu_3' : 'turquoise', \
				 'Bleu_4' : 'mediumpurple', 'METEOR' : 'dimgray', 'ROUGE_L' : 'red', \
				 'CIDEr' : 'orange', 'USE' : 'green'}

		handles = []
		for eval_name in self.eval_scores:
			handle, = plt.plot(
				self.eval_scores[eval_name], 
				color=color[eval_name], 
				label=eval_name
			)
			handles.append(handle)

		plot_name = self.output_name + '_scores.png'
		plt.legend(loc='best')
		plt.savefig(plot_name)
		plt.close()


	def train(self):

		training_steps = len(self.dataset_info['img2idx']['train']) // self.batch_size
		validation_steps = len(self.dataset_info['img2idx']['val']) // self.batch_size
		print('TRAINING STEPS=', training_steps, 'VALIDATION STEPS=', validation_steps)

		filepath = self.output_name + '-{epoch:02d}-{val_loss:.0f}.hdf5'
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

		callbacks_list = [checkpoint, early_stopping]

		for it in range(self.num_iterations):
			if it != 0 and it % self.eval_every == 0:
				self.evaluate()

			val_data_generator = get_data_chunks(
				split='val', 
				dataset_name=self.dataset_name,
				cnn_name=self.cnn_name,
				special_toks=self.special_toks,
				dataset_info=self.dataset_info, 
				chunk_size=self.batch_size
			)

			data_generator = get_data_chunks(
				split='train', 
			    dataset_name=self.dataset_name, 
		 	    cnn_name=self.cnn_name, 
			    special_toks=self.special_toks,
			    dataset_info=self.dataset_info, 
			    chunk_size=self.batch_size
			)

			history = self.model.fit_generator(
					generator=data_generator,
					epochs=1,
					verbose=1,
					callbacks=callbacks_list,
					steps_per_epoch=training_steps,
					validation_data=val_data_generator,
					validation_steps=validation_steps,
					workers=4
			)
			self.plot_loss(history)


