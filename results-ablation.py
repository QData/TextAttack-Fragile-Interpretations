DESC='''
Code to calculate results based on candidates/interpretations
'''

# import language_check
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from textattack.shared import AttackedText

import matplotlib.pyplot as plt
from scipy.stats import spearmanr 

import pickle
import argparse
import numpy as np
import math
import re
from tqdm import tqdm
from matplotlib import collections
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def com_scores(atts,enc):
	score = 0
	for i in range(len(enc)):
		score += atts[i]*i
	return score/len(enc)

def l2_scores(atts):
	return torch.norm(atts)


def diff(x,y):
	"""Returns the set of indices for which this and other_attacked_text
	have different words."""
	indices = set()
	w1 = x.words
	w2 = y.words
	for i in range(min(len(w1), len(w2))):
		if w1[i] != w2[i]:
			indices.add(i)
	return indices



def rank_correlation(int1,int2):
	return spearmanr(int1.cpu().numpy().tolist(),int2.cpu().numpy().tolist())

def topk_intersection(int1,int2):
	k = int((int1.size()[0])/2)
	i1 = torch.argsort(torch.abs(int1),descending=True).cpu().numpy().tolist()[:k]
	i2 = torch.argsort(torch.abs(int2),descending=True).cpu().numpy().tolist()[:k]
	return len([x for x in i1 if x in i2])/k


def plot_violin_graph_rand(data_to_plot,data_to_plot_rand, xlabel, ylabel,nm,met):
	fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), sharey=True)
	# print(data_to_plot)
	f = ax1.violinplot(data_to_plot,showmeans=True)
	f_rand = ax1.violinplot(data_to_plot_rand,showmeans=True)

	labels = []
	def add_label(violin, label):
		import matplotlib.patches as mpatches
		color = violin["bodies"][0].get_facecolor().flatten()
		labels.append((mpatches.Patch(color=color), label))

	add_label(f,"Metric:"+met)
	add_label(f_rand,"Random")

	def set_axis_style(ax, labels):
		ax.get_xaxis().set_tick_params(direction='out')
		ax.xaxis.set_ticks_position('bottom')
		ax.set_xticks(np.arange(1, len(labels) + 1))
		ax.set_xticklabels(labels)
		ax.set_xlim(0.25, len(labels) + 0.75)
		ax.tick_params(axis='both', which='major', labelsize=14)
		ax.set_xlabel(xlabel, fontsize=20)
		ax.set_ylabel(ylabel, fontsize=20)


	set_axis_style(ax1,["{0:.1f}".format(0.1*(i-1))+"-"+"{0:.1f}".format(0.1*i) for i in range(1,len(data_to_plot)+1)])

	plt.legend(*zip(*labels), loc=2)

	l = f['cmeans'].get_segments()
	lines = [np.array([0,1])]
	for i in range(len(l)):
		lines.append((l[i][0]+l[i][1])/2)

	l_rand = f_rand['cmeans'].get_segments()
	lines_rand = [np.array([0,1])]
	for i in range(len(l_rand)):
		lines_rand.append((l_rand[i][0]+l_rand[i][1])/2)

	w = collections.LineCollection([lines])
	w_rand = collections.LineCollection([lines_rand],color="tab:orange")
	ax1.add_collection(w)
	ax1.add_collection(w_rand)
	ax1.autoscale()


	# ax1.legend(proxies, ['Selection based on ExplainFooler-'+str(metric), 'Random Selection'])

	plt.savefig(nm)

def process_string(string):
	string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
	string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
	# U . S . -> U.S.
	string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
	# reduce left space
	string = re.sub("( )([,\.!?:;)])", r"\2", string)
	# reduce right space
	string = re.sub("([(])( )", r"\1", string)
	string = re.sub("s '", "s'", string)
	# reduce both space
	string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
	string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
	string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
	string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
	# string = re.sub(" ' ", "'", string)
	return string

def get_ppl(texts):
	ppl_model = GPT2LMHeadModel.from_pretrained('gpt2').cuda() 
	ppl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	ppl_model.eval()
	eval_loss = 0
	nb_eval_steps = 0

	with torch.no_grad():
		for text in texts:
			text = process_string(text)
			input_ids = torch.tensor(ppl_tokenizer.encode(text, add_special_tokens=True))
			if len(input_ids) < 2:
				continue
			input_ids = input_ids.cuda()
			outputs = ppl_model(input_ids, labels=input_ids)
			lm_loss = outputs[0]
			eval_loss += lm_loss.mean().item()
			# print(eval_loss)
			nb_eval_steps += 1

	eval_loss = eval_loss / nb_eval_steps
	perplexity = torch.exp(torch.tensor(eval_loss))

	return perplexity.item()



def main():
	parser=argparse.ArgumentParser(description=DESC, formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-m","--model",required=True, help="Name of model")
	parser.add_argument("-d","--dataset",required=True, help="Name of dataset")
	parser.add_argument("-s","--split",required=True, help="Split of dataset")
	parser.add_argument("-num","--number",required=False, type=int, default=-1, help="Number of samples from dataset")
	parser.add_argument("-mf","--modelfolder",required=False, default='./models/',help="Folder to load models from")
	parser.add_argument("-if","--interpretfolder",required=False, default='./interpretations/',help="Folder to store interpretations")
	parser.add_argument("-im","--interpretmethod",required=True,help="Interpretation Method (IG/LIME)")
	parser.add_argument("-rf","--resultfolder",required=False,default="./results/",help="Folder to store results")
	args = parser.parse_args()

	global model
	global tokenizer

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

	model.eval()



	interpretation_name = args.interpretfolder+"interpretations-"+args.dataset+"-"+args.model+"-"+args.interpretmethod+'-'+str(args.number)+".pkl"
	original_interpretation_name = args.interpretfolder+"original_sentences/original-interpretations-"+args.dataset+"-"+args.model+"-"+args.interpretmethod+'-'+str(args.number)+".pkl"
	
	with open(interpretation_name, 'rb') as f:
		interp = pickle.load(f)

	with open(original_interpretation_name, 'rb') as f:
		orig_interp = pickle.load(f)

	
	if args.number == -1:
		args.number = len(fin)

	lom_list = []
	l2_list = []

	lom_list_rand = []
	l2_list_rand = []

	ids = range(args.number)
	for fn,idx in zip(interp,ids):
		try:
			print("Calculating on %d of %d total sentences" %(idx,len(ids)))
			p2s = {}
			p2s_rand ={}
			t2s = {}
			t2s_rand ={}
			cms_original = com_scores(orig_interp[idx][1][0],orig_interp[idx][1][1])
			for i in range(len(fn)):
				try:
					r = random.randint(0,len(fn[i])-1)
					cms_rand = com_scores(fn[i][r][1][0],fn[i][r][1][1])

					for j in range(len(fn[i])):
							w = len(diff(AttackedText(orig_interp[idx][0]),AttackedText(fn[i][j][0])))

							if w not in p2s_rand:						
								p2s_rand[w] = (np.linalg.norm(cms_original-cms_rand),fn[i][r])

							if w not in t2s_rand:						
								t2s_rand[w] = (np.linalg.norm(cms_original-cms_rand),fn[i][r])


							cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])
							
							if w not in p2s:						
								p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])

							else:
								if np.linalg.norm(cms_original-cms) > p2s[w][0]:
									p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])

							if w not in t2s:
								t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
							else:
								if l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0])> t2s[w][0]:
									t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])

				except:
					print("Skipped sentence #",idx)
					pass

			lom_list.append(p2s)
			l2_list.append(t2s)
			lom_list_rand.append(p2s_rand)
			l2_list_rand.append(t2s_rand)
		except:
			print("Fail on sentence: %d"%idx)
			lom_list.append(None)
			l2_list.append(None)
			lom_list_rand.append(None)
			l2_list_rand.append(None)


	resultsx = []
	results_lom = []
	results_l2 = []

	results_lom_rand = []
	results_l2_rand = []


	for p,t,p_rand,t_rand,idx in zip(lom_list,l2_list,lom_list_rand,l2_list_rand,ids):
	# for p_rand,t_rand,idx in zip(lom_list_rand,l2_list_rand,ids):
		if p==None or t==None or p_rand==None or t_rand==None or orig_interp==None:
			continue
		# if p_rand==None or t_rand==None or orig_interp==None:
		# 	continue

		i0,f0 = orig_interp[idx][1][0],orig_interp[idx][1][1]
		sent_len = len(f0)-2

		for per_w in range(1,len(p_rand)):
			try:
				results_lom.append(rank_correlation( i0, p[per_w][1][1][0])[0] )
				results_l2.append(rank_correlation( i0, t[per_w][1][1][0])[0] )
				results_lom_rand.append(rank_correlation( i0, p_rand[per_w][1][1][0])[0] )
				results_l2_rand.append(rank_correlation( i0, t_rand[per_w][1][1][0])[0] )
				# results_lom.append(topk_intersection( i0, p[per_w][1][1][0]) )
				# results_l2.append(topk_intersection( i0, t[per_w][1][1][0]) )
				# results_lom_rand.append(topk_intersection( i0, p_rand[per_w][1][1][0]) )
				# results_l2_rand.append(topk_intersection( i0, t_rand[per_w][1][1][0]) )
				resultsx.append(per_w/sent_len)
			except:
				print("error")
				pass
		print(idx)




	d_lom= {}
	d_l2 = {}

	d_lom_rand= {}
	d_l2_rand = {}

	for ix in range(len(resultsx)):
		idex = int(10*resultsx[ix])
		if idex not in d_lom:
			d_lom[idex] = [max(0,results_lom[ix])]
		else:
			if results_lom[ix] < 0:
				d_lom[idex].append(0)
			else:
				d_lom[idex].append(results_lom[ix])

		if idex not in d_lom_rand:
			d_lom_rand[idex] = [max(0,results_lom_rand[ix])]
		else:
			if results_lom_rand[ix] < 0:
				d_lom_rand[idex].append(0)
			else:
				d_lom_rand[idex].append(results_lom_rand[ix])

		if idex not in d_l2:
			d_l2[idex] = [max(0,results_l2[ix])]
		else:
			if results_l2[ix] < 0:
				d_l2[idex].append(0)
			else:
				d_l2[idex].append(results_l2[ix])

		if idex not in d_l2_rand:
			d_l2_rand[idex] = [max(0,results_l2_rand[ix])]
		else:
			if results_l2_rand[ix] < 0:
				d_l2_rand[idex].append(0)
			else:
				d_l2_rand[idex].append(results_l2_rand[ix])



	data_to_plot_lom = [d_lom[i] for i in range(min(5,len(d_lom)))]
	data_to_plot_l2 = [d_l2[i] for i in range(min(5,len(d_l2)))]
	data_to_plot_lom_rand = [d_lom_rand[i] for i in range(min(5,len(d_lom_rand)))]
	data_to_plot_l2_rand = [d_l2_rand[i] for i in range(min(5,len(d_l2_rand)))]


	print("Rand Mean LOM:",[np.mean(m) for m in data_to_plot_lom_rand])
	print("Rand Mean L2:",[np.mean(m) for m in data_to_plot_l2_rand])

	t1 = [np.mean(m) for m in data_to_plot_lom_rand]
	t2 = [np.mean(m) for m in data_to_plot_l2_rand]

	for t in t1:
		print(np.round(t,2))

	print()

	# for t in t2:
	# 	print(np.round(t,2))

	plot_violin_graph_rand(data_to_plot_lom,data_to_plot_lom_rand,"Ratio of words perturbed","Rank Correlation (LOM)",args.resultfolder+"rank-corr-lom-rand-"+str(args.dataset)+".png",met="LOM")
	plot_violin_graph_rand(data_to_plot_l2,data_to_plot_l2_rand,"Ratio of words perturbed","Rank Correlation (L2)",args.resultfolder+"rank-corr-l2-rand-"+str(args.dataset)+".png",met="L2")


	# Only use this if you want original dataset stats
	dataset_ppl = []
	candidate_ppl_lom = []
	candidate_ppl_l2 = []

	grammar_check_lom = []

	for p,t,idx in zip(lom_list,l2_list,ids):
		if p==None or t==None or orig_interp==None:
			continue
		dataset_ppl.append(orig_interp[idx][0])

		for per_w in range(1,len(p)):
			try:
				candidate_ppl_lom.append(p[per_w][1][0])
				candidate_ppl_l2.append(t[per_w][1][0])
				resultsx.append(per_w/sent_len)
			except:
				print("error")
				pass
		print(idx)

	print(dataset_ppl[0])
	print(candidate_ppl_lom[0])
	print("Average dataset Perplexity:",get_ppl(dataset_ppl))
	print("Average LOM Perplexity:",get_ppl(candidate_ppl_lom))
	print("Average L2 Perplexity:",get_ppl(candidate_ppl_l2))



main()
			

