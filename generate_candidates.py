DESC='''
Function to generate all candidates usin ExplainFooler for a given model, dataset and split'
Requires custom Textattack installation
'''
import torch
from copy import deepcopy
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.shared import AttackedText

import tensorflow as tf
import pickle
import argparse
import logging

tf.enable_eager_execution()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
	parser=argparse.ArgumentParser(description=DESC, formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-m","--model",required=True, help="Name of model")
	parser.add_argument("-d","--dataset",required=True, help="Name of dataset")
	parser.add_argument("-s","--split",required=True, help="Split of dataset")
	parser.add_argument("-num","--number",required=True, type=int, help="Number of samples from dataset")
	parser.add_argument("-c","--candidatefolder",required=False, default='./candidates/',help="Folder to store candidates")
	parser.add_argument("-ld","--loadfromfolder",required=True, default=False, help="Whether to load data from - file or huggingface")
	parser.add_argument("-mf","--modelfolder",required=False, default='./models/',help="Folder to load models from")
	args = parser.parse_args()

	if args.model == "distilbert":
		if args.dataset == "sst2":
			model = AutoModelForSequenceClassification.from_pretrained(args.modelfolder+"distilbert-base-uncased-SST-2-glue^sst2-2021-01-11-09-08-54-383533").to(device)
			tokenizer = AutoTokenizer.from_pretrained(args.modelfolder+"distilbert-base-uncased-SST-2-glue^sst2-2021-01-11-09-08-54-383533")
		elif args.dataset == "agnews":
			model = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-uncased-ag-news")
			tokenizer = AutoTokenizer.from_pretrained("textattack/distilbert-base-uncased-ag-news")
		elif args.dataset == "imdb":
			model = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-uncased-imdb")
			tokenizer = AutoTokenizer.from_pretrained("textattack/distilbert-base-uncased-imdb")

	elif args.model == "roberta":
		if args.dataset == "sst2":
			model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
			tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
		elif args.dataset == "agnews":
			model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-ag-news")
			tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-ag-news")
		elif args.dataset == "imdb":
			model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-imdb")
			tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-imdb")

	elif args.model == "bert-adv":
		if args.dataset == "sst2":
			model = AutoModelForSequenceClassification.from_pretrained(args.modelfolder+"bert-sst2-adv")
			tokenizer = AutoTokenizer.from_pretrained(args.modelfolder+"bert-sst2-adv")

		elif args.dataset == "agnews":
			model = AutoModelForSequenceClassification.from_pretrained(args.modelfolder+"bert-ag-adv")
			tokenizer = AutoTokenizer.from_pretrained(args.modelfolder+"bert-ag-adv")

		elif args.dataset == "imdb":
			model = AutoModelForSequenceClassification.from_pretrained(args.modelfolder+"bert-imdb-adv")
			tokenizer = AutoTokenizer.from_pretrained(args.modelfolder+"bert-imdb-adv")


	if args.dataset == "sst2":
		ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
	elif args.dataset == "agnews":
		ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
	elif args.dataset == "imdb":
		ta_dataset = HuggingFaceDataset("imdb", split=args.split)


	ta_model = HuggingFaceModelWrapper(model, tokenizer)
	attack = TextFoolerJin2019.build(ta_model,1)
	results_iterable = attack.attack_dataset(ta_dataset, indices=range(args.number))
	fin = []
	for n,result in enumerate(results_iterable):
		print("Sample Number:",n)
		fin.append(result)

	dump_file_name = str(args.candidatefolder)+"candidates"+"-"+str(args.dataset)+"-"+str(args.model)+".pkl" 

	with open(dump_file_name, 'wb') as f:
		pickle.dump(fin, f)

main()

