# Andreea Musat, May 2018

import sys
import argparse
import tensorflow as tf

from utils import exit
from new_parser import load_dataset_info
from captions_generator import CaptionsGenerator
from keras.backend.tensorflow_backend import set_session

def main(args):
	special_toks = {}
	special_toks['start'] = '<start_tok>'
	special_toks['end'] = '<end_tok>'
	special_toks['unknown'] = '<unknown_tok>'

	dataset_info = load_dataset_info(args.dataset_name, args.cnn_name, special_toks)
	captions_generator = CaptionsGenerator(args, dataset_info)
	captions_generator.train()


def sanity_checks(args):

	try:
		assert args.dataset_name is not None
	except:
		exit('[ERROR] Dataset unknown.Exiting.')

	try:
		assert args.cnn_name is not None and \
			   args.cnn_name in ['googlenet', 'vgg-16']
	except:
		exit('[ERROR] CNN name unknown. Exiting.')

	try:
		assert args.model_name is not None
	except:
		exit('[ERROR] Language model name unknown. Exiting.')

	try:
		assert args.learning_rate is not None and \
			   float(args.learning_rate) >= 0 and \
			   float(args.learning_rate) <= 1
	except:
		exit('[ERROR] Learning rate should be in [0, 1]')

	try:
		assert args.batch_size is not None
	except:
		exit('[ERROR] Batch size unknown')

	try:
		assert args.embed_size is not None
	except:
		exit('[ERROR] Embed size unknown. Exiting.')

	args.batch_size = int(args.batch_size)
	args.embed_size = int(args.embed_size)
	args.eval_every = int(args.eval_every)
	args.num_iterations = int(args.num_iterations)
	args.learning_rate = float(args.learning_rate)


def limit_gpu_usage(args):
	if not args.use_gpu:
		return

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	set_session(tf.Session(config=config))		
	print('Set GPU usage percent to 0.1')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset-name', dest='dataset_name')
	parser.add_argument('--cnn-name', dest='cnn_name')
	parser.add_argument('--model-name', dest='model_name')
	parser.add_argument('--learning-rate', dest='learning_rate', type=float)
	parser.add_argument('--batch-size', dest='batch_size', type=int)
	parser.add_argument('--embed-size', dest='embed_size', type=int)
	parser.add_argument('--eval-every', dest='eval_every', type=int, default=1)
	parser.add_argument('--num-iterations', dest='num_iterations', type=int)
	parser.add_argument('--use-gpu', dest='use_gpu', type=bool, default=False)
	parser.add_argument('--plot-name', dest='plot_name')
	parser.add_argument('--use-beam-search', dest='use_beam_search', type=bool, default=False)
	
	args, unknown = parser.parse_known_args()

	sanity_checks(args)
	limit_gpu_usage(args)
	main(args)