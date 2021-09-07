DESC='''
Code to calculate the interpretations using IG or LIME for all candidates 
for a given model,dataset and storing for future use
'''

import warnings
warnings.filterwarnings("ignore")

import torch
from copy import deepcopy
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from captum.attr import LayerGradientXActivation
from captum.attr import visualization as viz
from lime.lime_text import LimeTextExplainer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.shared import AttackedText

import matplotlib.pyplot as plt
from scipy.stats import spearmanr 

import pickle
import argparse
import numpy as np
from matplotlib import collections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = None
tokenizer = None


def get_interpret_ig(x,l,funct,lig):
	inp = torch.LongTensor([tokenizer.encode(x,truncation=True)]).to(device)
	label = torch.LongTensor([l]).to(device)
	bsl = torch.LongTensor([0]*inp.size()[1]).unsqueeze(0).to(device)
	attributions,delta = lig.attribute(inputs=inp,
								  baselines=bsl,
								  # layer_baselines=torch.Tensor([0]),
								  n_steps = 50,
								  target = label,
								  return_convergence_delta=True
								  )

	atts = attributions.sum(dim=-1).squeeze(0)
	atts = atts / torch.norm(atts)
	
	f = tokenizer.convert_ids_to_tokens(tokenizer.encode(x,truncation=True))
	f, atts = funct(f,atts.detach().cpu().numpy().tolist())

	return atts,f

def get_interpret_lime(inp,l=None,explainer=None):
	exp = explainer.explain_instance(inp, calculate_lime, num_samples=500, num_features=500)
	words = exp.as_list()[:]
	att = []
	for i in exp.as_map():
		att = sorted(exp.as_map()[i])
	
	att = [i[1] for i in att]
	att = torch.Tensor(att)
		
	maps = exp.as_map()[1]
	# print(words)
	m = {}
	for i in range(len(maps)):
		m[maps[i][0]] = words[i][0] 
	w = [0]*len(m)
	for i in m:
		w[i] = m[i]
	assert len(w) == len(att)
	return att,w

def combine_words_distilbert_average(broken,atts):
	formed = []
	i = 0
	at = []
	while i < len(broken):
		j = i+1
		temp = broken[i]
		cnt=1
		at.append(atts[i])
		while j < len(broken) and (broken[j]).startswith("#"):
			temp += broken[j].split("##")[-1]
			at[-1] += atts[j]
			j+=1
			cnt+=1
		at[-1] /= cnt
		formed.append(temp)
		i = j
	assert len(at) == len(formed)
	return formed,torch.Tensor(at)

def combine_words_disilbert_minmax(broken,atts):
	formed = []
	i = 0
	at = []
	while i < len(broken):
		j = i+1
		temp = broken[i]
		cnt=1
		at.append(atts[i])
		track = [atts[i]]
		while j < len(broken) and (broken[j]).startswith("#"):
			temp += broken[j].split("##")[-1]
			# at[-1] += atts[j]
			track.append(atts[j])
			j+=1
			cnt+=1
		at[-1] = obj(track)
		formed.append(temp)
		i = j
	assert len(at) == len(formed)
	return formed,torch.Tensor(at)

def combine_words_roberta_average(broken,atts):
	formed = []
	i = 0
	at = []
	while i < len(broken):
		j = i+1
		temp = broken[i].split("Ġ")[-1]
		cnt=1
		at.append(atts[i])
		while j < len(broken) and not (broken[j]).startswith("Ġ"):
			temp += broken[j]
			at[-1] += atts[j]
			j+=1
			cnt+=1
		at[-1] /= cnt
		formed.append(temp)
		i = j
	assert len(at) == len(formed)
	return formed,torch.Tensor(at).to(device)

def combine_words_roberta_minmax(broken,atts):
	formed = []
	i = 0
	at = []
	while i < len(broken):
		j = i+1
		temp = broken[i].split("Ġ")[-1]
		cnt=1
		at.append(atts[i])
		track = [atts[i]]
		while j < len(broken) and not (broken[j]).startswith("Ġ"):
			temp += broken[j]
			# at[-1] += atts[j]
			track.append(atts[j])
			j+=1
			cnt+=1
		# at[-1] /= cnt
		at[-1] = obj(track)
		formed.append(temp)
		i = j
	assert len(at) == len(formed)
	return formed,torch.Tensor(at)

def combine_words_lime_roberta_break(broken):
	formed = []
	i = 0
	at = []
	while i < len(broken):
		j = i+1
		temp = broken[i].split("Ġ")[-1]
		cnt=1
		while j < len(broken) and not (broken[j]).startswith("Ġ"):
			temp += broken[j]
			j+=1
			cnt+=1
		formed.append(temp)
		i = j
	return formed

def combine_words_lime_distil_break(broken):
	formed = []
	i = 0
	at = []
	while i < len(broken):
		j = i+1
		temp = broken[i]
		cnt=1
		while j < len(broken) and (broken[j]).startswith("#"):
			temp += broken[j].split("##")[-1]
			j+=1
			cnt+=1
		formed.append(temp)
		i = j
	return formed

def obj(track):
	mx = max(track)
	mn = min(track)

	if abs(mn)>abs(mx):
		return mn
	else:
		return mx

def break_string(x):
	f = tokenizer.tokenize(x)
	return f

def breaker_distilbert(x):
	x = x.lower()
	p = combine_words_lime_distil_break(break_string(x))
	return p

def breaker_roberta(x):
	x = x.lower()
	p = combine_words_lime_roberta_break(break_string(x))
	return p

def forward_lig(x):
    return model(x)[0]

def calculate_lime(inp):
	labels = []
	for i in inp:
		labels.append(model(torch.LongTensor([tokenizer.encode(i,truncation=True)]).to(device))[0].detach().cpu().numpy()[0])
	return torch.Tensor(labels).cpu().numpy()

def main():
	parser=argparse.ArgumentParser(description=DESC, formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-m","--model",required=True, help="Name of model")
	parser.add_argument("-d","--dataset",required=True, help="Name of dataset")
	parser.add_argument("-s","--split",required=True, help="Split of dataset")
	parser.add_argument("-num","--number",required=False, type=int, default=-1, help="Number of samples from dataset")
	parser.add_argument("-c","--candidatefolder",required=False, default='./candidates/',help="Folder to load candidates")
	parser.add_argument("-mf","--modelfolder",required=False, default='./models/',help="Folder to load models from")
	parser.add_argument("-if","--interpretfolder",required=False, default='./interpretations/',help="Folder to store interpretations")
	parser.add_argument("-im","--interpretmethod",required=True,help="Interpretation Method (IG/LIME)")
	parser.add_argument("-oif","--originalinterpretfolder",required=False,default='./interpretations/original_sentences/',help="Folder to store original interpretations")
	args = parser.parse_args()

	global model
	global tokenizer

	if args.model == "distilbert":
		if args.dataset == "sst2":
			model = AutoModelForSequenceClassification.from_pretrained(args.modelfolder+"distilbert-base-uncased-SST-2-glue^sst2-2021-01-11-09-08-54-383533")
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

	model.to(device)
	model.eval()


	if args.dataset == "sst2":
		ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
		class_names = ['positive', 'negative']
		keyword = "sentence"

	elif args.dataset == "agnews":
		ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
		class_names = ["World", "Sports","Business","Sci/Tech"]
		keyword = "text"

	elif args.dataset == "imdb":
		ta_dataset = HuggingFaceDataset("imdb", split=args.split)
		class_names = ['positive', 'negative']
		keyword = "text"

	if args.interpretmethod == "IG":
		get_interpret = get_interpret_ig
		if args.model == "distilbert":
			lig = LayerIntegratedGradients(forward_lig, model.distilbert.embeddings)
			comb_word_func = combine_words_disilbert_minmax

		elif args.model == "roberta":
			lig = LayerIntegratedGradients(forward_lig, model.roberta.embeddings)
			comb_word_func = combine_words_roberta_minmax

		elif args.model == "bert-adv":
			lig = LayerIntegratedGradients(forward_lig, model.bert.embeddings)
			comb_word_func = combine_words_disilbert_minmax

	elif args.interpretmethod == "LIME":
		get_interpret = get_interpret_lime
		if args.model == "distilbert":
			explainer = LimeTextExplainer(class_names=class_names, split_expression = breaker_distilbert)
		elif args.model == "roberta":
			explainer = LimeTextExplainer(class_names=class_names, split_expression = breaker_roberta)
		elif args.model == "bert-adv":
			explainer = LimeTextExplainer(class_names=class_names, split_expression = breaker_distilbert)


	candidate_name = str(args.candidatefolder)+"candidates-"+str(args.dataset)+"-"+str(args.model)+'.pkl'
	print("Loading from: ",candidate_name)

	with open(candidate_name, 'rb') as f:
		fin = pickle.load(f)

	if args.number == -1:
		args.number = len(fin)

	original_dataset_interpret = []
	save_list = []
	ids = range(args.number)
	for fn,idx in zip(fin,ids):
		print("Calculating on %d of %d total sentences" %(idx,len(ids)))
		try:
			s = []
			original_interpret = None
			for i in range(len(fn)):
				c = []
				for j in range(len(fn[i])):
					if torch.argmax(model(torch.LongTensor([tokenizer.encode(fn[i][j].attacked_text.text,truncation=True)]).to(device))[0][0]).item() == ta_dataset[idx][1]:
						try:
							if args.interpretmethod == "IG":
								inter = get_interpret_ig(fn[i][j].attacked_text.text.lower(),ta_dataset[idx][1],comb_word_func,lig)
								if original_interpret == None:
									original_interpret = get_interpret_ig(ta_dataset[idx][0][keyword].lower(),ta_dataset[idx][1],comb_word_func,lig)
							elif args.interpretmethod == "LIME":
								inter = get_interpret_lime(fn[i][j].attacked_text.text.lower(),ta_dataset[idx][1],explainer)
								if original_interpret == None:	
									original_interpret = get_interpret_lime(ta_dataset[idx][0][keyword].lower(),ta_dataset[idx][1],explainer)

						except:
							inter=None
							print("error on a candidate, skipped..")
							pass
						print("Calculated %d of %d candidates"%(j,len(fn[i])))
						print("Calculating on %d of %d total sentences" %(idx,len(ids)))
						c.append([fn[i][j].attacked_text.text,inter])
				s.append(c)
			save_list.append(s)
			original_dataset_interpret.append([ta_dataset[idx][0][keyword],original_interpret])
			original_interpret = None
		except:
			print("Sentence Skipped completely: ",idx)
			save_list.append([])
			original_dataset_interpret.append([])

	print(save_list)
	print(original_dataset_interpret)
	interpretation_name = args.interpretfolder+'interpretations-'+str(args.dataset)+'-'+str(args.model)+'-'+args.interpretmethod+"-"+str(args.number)+'.pkl'
	original_interpretation_name = args.originalinterpretfolder+'original-interpretations-'+str(args.dataset)+'-'+str(args.model)+'-'+args.interpretmethod+"-"+str(args.number)+'.pkl'
	print("Saving to: ",interpretation_name)
	with open(interpretation_name, 'wb') as f:
		pickle.dump(save_list,f)
	with open(original_interpretation_name, 'wb') as f:
		pickle.dump(original_dataset_interpret,f)


main()
			

