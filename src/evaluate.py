# Andreea Musat, May 2018

import os, sys

for metric_name in ['bleu', 'cider', 'meteor', 'rouge']:
	print('appended ', os.path.join('.', 'pycocoevalcap', metric_name))
	sys.path.append(os.path.join('.', 'pycocoevalcap', metric_name))

from bleu import Bleu
from cider import Cider
from meteor import Meteor
from rouge import Rouge
# from spice import Spice

def eval_captions(gt_captions, res_captions):
	"""
		gt_captions = ground truth captions; 5 per image
		res_captions = captions generated by the model to be evaluated
	"""
	print('ground truth captions')
	print(gt_captions)

	print('RES CAPTIONS')
	print(res_captions)

	scorers = [
		(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
		(Meteor(),"METEOR"),
		(Rouge(), "ROUGE_L"),
		(Cider(), "CIDEr"), 
	]

	res = []
	
	for scorer, method in scorers:
		print('computing %s score...' % (scorer.method()))
		score, scores = scorer.compute_score(gt_captions, res_captions)
		if type(method) == list:
			for sc, scs, m in zip(score, scores, method):
				print("%s: %0.3f"%(m, sc))
				res.append((m, sc))
		else:
				print("%s: %0.3f"%(method, score))
				res.append((method, score))

	return res
