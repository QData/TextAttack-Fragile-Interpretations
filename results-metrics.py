DESC='''
Code to calculate results based on candidates/interpretations
'''

# import language_check
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.nn import Softmax
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from textattack.shared import AttackedText
from textattack.datasets import HuggingFaceDataset


import matplotlib.pyplot as plt
from scipy.stats import spearmanr 

import pickle
import argparse
import numpy as np
import math
import re
from tqdm import tqdm
from matplotlib import collections
import language_tool_python

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


def plot_violin_graph(data_to_plot, xlabel, ylabel,nm):
	fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), sharey=True)
	# print(data_to_plot)
	f = ax1.violinplot(data_to_plot,showmeans=True)

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

	l = f['cmeans'].get_segments()
	lines = [np.array([0,1])]
	for i in range(len(l)):
		lines.append((l[i][0]+l[i][1])/2)

	w = collections.LineCollection([lines])
	ax1.add_collection(w)
	ax1.autoscale()
	plt.savefig(nm)

# Taken from CLARE
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
		for text in tqdm(texts):
			text = process_string(text)
			input_ids = torch.tensor(ppl_tokenizer.encode(text, add_special_tokens=True,truncation=True))
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


def get_grm_err(dataset_grm,candidate_grm):
	tool = language_tool_python.LanguageTool('en-US')
	grammar_diffs = []
	errs = 0
	for i in tqdm(range(len(dataset_grm))):
		ref = len(tool.check(process_string(dataset_grm[i])))
		for j in range(len(candidate_grm[i])):
			hypo = len(tool.check(process_string(candidate_grm[i][j])))
			# print(hypo)
			grammar_diffs.append(np.abs(hypo-ref))
	
	return np.mean(grammar_diffs)


def rationalize(w0,i0,w1,i1):
	w_list0= {}
	w_list1= {}
	# orig = {}
	for i in range(len(w0)):
		w_list0[w0[i]] = i
	for i in range(len(w1)):
		w_list1[w1[i]] = i

	pos1 = []
	for i in w_list0:
		if i not in w_list1:
			pos1.append(w_list0[i])

	pos2 = []
	for i in w_list1:
		if i not in w_list0:
			pos2.append(w_list1[i])

	pos = list(set(pos1).intersection(pos2))

	temp = i1.numpy().tolist()
	t = len(i0.numpy().tolist())-len(i1.numpy().tolist())
	if t>0:
		for p in range(t):
			temp.insert(pos[p],0)
	elif t<0:
		for p in range(-t):
			temp.pop(pos[p])

	# assert len(temp) == len(i0.numpy().tolist())

	return w0,i0,w0,torch.Tensor(temp)





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
	parser.add_argument("-met","--metric",required=True,help="What metric to calculate - RC-TopK(rck)/Perplexity(ppl)/Grammar(grm)/Confidence(conf)")
	parser.add_argument("-c","--candidatefolder",required=False, default='./candidates/',help="Folder to store candidates")
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

	if args.metric == "rkc":

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

		ids = range(args.number)
		for fn,idx in zip(interp,ids):
			# if orig_interp[idx][1] == None:
			# 	print("Skippped due to external factors")
			# 	continue
			try:
			# print(orig_interp[idx])
				print("Calculating on %d of %d total sentences" %(idx,len(ids)))
				p2s = {}
				t2s = {}
				cms_original = com_scores(orig_interp[idx][1][0],orig_interp[idx][1][1])
				cntim = 0
				tot = 0
				for i in range(len(fn)):
					for j in range(len(fn[i])):
						try:
							w = len(diff(AttackedText(orig_interp[idx][0]),AttackedText(fn[i][j][0])))
							cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							w_o = orig_interp[idx][1][1]
							i_o = orig_interp[idx][1][0]
							w_c = fn[i][j][1][1]
							i_c = fn[i][j][1][0]

							if orig_interp[idx][1][0].size() != fn[i][j][1][0].size():
								
								w_o,i_o,w_c,i_c = rationalize(w_o,i_o,w_c,i_c)
								# orig_interp[idx][1][1],orig_interp[idx][1][0],fn[i][j][1][1],fn[i][j][1][0] = rationalize(w_o,i_o,w_c,i_c)
								cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							if w not in p2s:						
								# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
								p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])
							else:
								if np.linalg.norm(cms_original-cms) > p2s[w][0]:
									# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
									p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])

							if w not in t2s:
								# t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							else:
								# if l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0])> t2s[w][0]:
								# 	t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								if l2_scores(i_o-i_c)> t2s[w][0]:
									t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							tot+=1

						except:
							# print("Skipped sentence #",i,j)
							cntim+=1
							pass
				print("Skipped",cntim,tot)
				lom_list.append(p2s)
				l2_list.append(t2s)
			except:
				print("Fail on sentence: %d"%idx)
				lom_list.append(None)
				l2_list.append(None)



		resultsx = []
		results_lom = []
		results_l2 = []


		for p,t,idx in zip(lom_list,l2_list,ids):
			if p==None or t==None or orig_interp==None or orig_interp[idx][1]==None:
				print("Skipped,",idx)
				continue

			i0,f0 = orig_interp[idx][1][0],orig_interp[idx][1][1]
			sent_len = min(len(f0)-2,128)

			for per_w in range(1,len(p)):
				try:
					results_lom.append(rank_correlation( i0, p[per_w][1][1][0])[0] )
					results_l2.append(rank_correlation( i0, t[per_w][1][1][0])[0] )
					# results_lom.append(topk_intersection( i0, p[per_w][1][1][0]) )
					# results_l2.append(topk_intersection( i0, t[per_w][1][1][0]) )
					resultsx.append(per_w/sent_len)
				except:
					print("error")
					pass
			print(idx)




		d_lom= {}
		d_l2 = {}
		for ix in range(len(resultsx)):
			idex = int(10*resultsx[ix])
			if idex not in d_lom:
				d_lom[idex] = [max(0,results_lom[ix])]
			else:
				if results_lom[ix] < 0:
					d_lom[idex].append(0)
				else:
					d_lom[idex].append(results_lom[ix])

			if idex not in d_l2:
				d_l2[idex] = [max(0,results_l2[ix])]
			else:
				if results_l2[ix] < 0:
					d_l2[idex].append(0)
				else:
					d_l2[idex].append(results_l2[ix])


		data_to_plot_lom = [d_lom[i] for i in range(min(5,len(d_lom)))]
		data_to_plot_l2 = [d_l2[i] for i in range(min(5,len(d_l2)))]

		print("Mean LOM:",[np.mean(m) for m in data_to_plot_lom])
		print("Mean L2:",[np.mean(m) for m in data_to_plot_l2])

		plot_violin_graph(data_to_plot_lom,"Ratio of words perturbed","Rank Correlation(LOM)",args.resultfolder+"violinplot-"+str(args.dataset)+"-"+str(args.model)+"-"+"rank-corr-lom.png")
		plot_violin_graph(data_to_plot_l2,"Ratio of words perturbed","Rank Correlation(L2)",args.resultfolder+"violinplot-"+str(args.dataset)+"-"+str(args.model)+"-"+"rank-corr-l2.png")

	if args.metric == "topk":

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

		ids = range(args.number)
		for fn,idx in zip(interp,ids):
			# if orig_interp[idx][1] == None:
			# 	print("Skippped due to external factors")
			# 	continue
			try:
			# print(orig_interp[idx])
				print("Calculating on %d of %d total sentences" %(idx,len(ids)))
				p2s = {}
				t2s = {}
				cms_original = com_scores(orig_interp[idx][1][0],orig_interp[idx][1][1])
				cntim = 0
				tot = 0
				for i in range(len(fn)):
					for j in range(len(fn[i])):
						try:
							w = len(diff(AttackedText(orig_interp[idx][0]),AttackedText(fn[i][j][0])))
							cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							w_o = orig_interp[idx][1][1]
							i_o = orig_interp[idx][1][0]
							w_c = fn[i][j][1][1]
							i_c = fn[i][j][1][0]

							if orig_interp[idx][1][0].size() != fn[i][j][1][0].size():
								
								w_o,i_o,w_c,i_c = rationalize(w_o,i_o,w_c,i_c)
								# orig_interp[idx][1][1],orig_interp[idx][1][0],fn[i][j][1][1],fn[i][j][1][0] = rationalize(w_o,i_o,w_c,i_c)
								cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							if w not in p2s:						
								# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
								p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])
							else:
								if np.linalg.norm(cms_original-cms) > p2s[w][0]:
									# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
									p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])

							if w not in t2s:
								# t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							else:
								# if l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0])> t2s[w][0]:
								# 	t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								if l2_scores(i_o-i_c)> t2s[w][0]:
									t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							tot+=1

						except:
							# print("Skipped sentence #",i,j)
							cntim+=1
							pass
				print("Skipped",cntim,tot)
				lom_list.append(p2s)
				l2_list.append(t2s)
			except:
				print("Fail on sentence: %d"%idx)
				lom_list.append(None)
				l2_list.append(None)



		resultsx = []
		results_lom = []
		results_l2 = []


		for p,t,idx in zip(lom_list,l2_list,ids):
			if p==None or t==None or orig_interp==None or orig_interp[idx][1]==None:
				print("Skipped,",idx)
				continue

			i0,f0 = orig_interp[idx][1][0],orig_interp[idx][1][1]
			sent_len = min(len(f0)-2,128)

			for per_w in range(1,len(p)):
				try:
					# results_lom.append(rank_correlation( i0, p[per_w][1][1][0])[0] )
					# results_l2.append(rank_correlation( i0, t[per_w][1][1][0])[0] )
					results_lom.append(topk_intersection( i0, p[per_w][1][1][0]) )
					results_l2.append(topk_intersection( i0, t[per_w][1][1][0]) )
					resultsx.append(per_w/sent_len)
				except:
					print("error")
					pass
			print(idx)




		d_lom= {}
		d_l2 = {}
		for ix in range(len(resultsx)):
			idex = int(10*resultsx[ix])
			if idex not in d_lom:
				d_lom[idex] = [max(0,results_lom[ix])]
			else:
				if results_lom[ix] < 0:
					d_lom[idex].append(0)
				else:
					d_lom[idex].append(results_lom[ix])

			if idex not in d_l2:
				d_l2[idex] = [max(0,results_l2[ix])]
			else:
				if results_l2[ix] < 0:
					d_l2[idex].append(0)
				else:
					d_l2[idex].append(results_l2[ix])


		data_to_plot_lom = [d_lom[i] for i in range(min(5,len(d_lom)))]
		data_to_plot_l2 = [d_l2[i] for i in range(min(5,len(d_l2)))]

		print("Mean LOM:",[np.mean(m) for m in data_to_plot_lom])
		print("Mean L2:",[np.mean(m) for m in data_to_plot_l2])

		plot_violin_graph(data_to_plot_lom,"Ratio of words perturbed","TopK (LOM)",args.resultfolder+"violinplot-"+str(args.dataset)+"-"+str(args.model)+"-"+"rank-corr-lom.png")
		plot_violin_graph(data_to_plot_l2,"Ratio of words perturbed","TopK (L2)",args.resultfolder+"violinplot-"+str(args.dataset)+"-"+str(args.model)+"-"+"rank-corr-l2.png")

	if args.metric == "ppl":

		if args.dataset == "sst2":
			ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
			keyword = "sentence"
		elif args.dataset == "agnews":
			ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
			keyword = "text"
		elif args.dataset == "imdb":
			ta_dataset = HuggingFaceDataset("imdb", split=args.split)
			keyword = "text"

		candidate_name = str(args.candidatefolder)+"candidates"+"-"+str(args.dataset)+"-"+str(args.model)+".pkl"

		with open(candidate_name, 'rb') as f:
			candidates = pickle.load(f)

		if args.number == -1:
			args.number = len(candidates)

		dataset_ppl = []
		candidate_ppl = []

		for idx in range(args.number):
			dataset_ppl.append(ta_dataset[idx][0][keyword])
			for i in range(len(candidates[idx])):
				for j in range(len(candidates[idx][i])):
					candidate_ppl.append(candidates[idx][i][j].attacked_text.text.lower())
			print(idx)

		dataset_ppl_num = get_ppl(dataset_ppl)
		candidate_ppl_num = get_ppl(candidate_ppl)
		print("Average dataset Perplexity:",dataset_ppl_num)
		print("Average LOM Perplexity:",candidate_ppl_num)

	if args.metric == "ppl-selected":

		if args.dataset == "sst2":
			ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
			keyword = "sentence"
		elif args.dataset == "agnews":
			ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
			keyword = "text"
		elif args.dataset == "imdb":
			ta_dataset = HuggingFaceDataset("imdb", split=args.split)
			keyword = "text"

		interpretation_name = args.interpretfolder+"interpretations-"+args.dataset+"-"+args.model+"-"+args.interpretmethod+'-'+str(args.number)+".pkl"
		original_interpretation_name = args.interpretfolder+"original_sentences/original-interpretations-"+args.dataset+"-"+args.model+"-"+args.interpretmethod+'-'+str(args.number)+".pkl"
		
		with open(interpretation_name, 'rb') as f:
			interp = pickle.load(f)

		with open(original_interpretation_name, 'rb') as f:
			orig_interp = pickle.load(f)

		candidate_name = str(args.candidatefolder)+"candidates"+"-"+str(args.dataset)+"-"+str(args.model)+".pkl"

		with open(candidate_name, 'rb') as f:
			candidates = pickle.load(f)

		if args.number == -1:
			args.number = len(candidates)

		dataset_ppl = []
		candidate_ppl = []

		if args.number == -1:
			args.number = len(fin)

		lom_list = []
		l2_list = []

		ids = range(args.number)
		for fn,idx in zip(interp,ids):
			# if orig_interp[idx][1] == None:
			# 	print("Skippped due to external factors")
			# 	continue
			try:
			# print(orig_interp[idx])
				print("Calculating on %d of %d total sentences" %(idx,len(ids)))
				p2s = {}
				t2s = {}
				cms_original = com_scores(orig_interp[idx][1][0],orig_interp[idx][1][1])
				cntim = 0
				tot = 0
				for i in range(len(fn)):
					for j in range(len(fn[i])):
						try:
							w = len(diff(AttackedText(orig_interp[idx][0]),AttackedText(fn[i][j][0])))
							cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							w_o = orig_interp[idx][1][1]
							i_o = orig_interp[idx][1][0]
							w_c = fn[i][j][1][1]
							i_c = fn[i][j][1][0]

							if orig_interp[idx][1][0].size() != fn[i][j][1][0].size():
								
								w_o,i_o,w_c,i_c = rationalize(w_o,i_o,w_c,i_c)
								# orig_interp[idx][1][1],orig_interp[idx][1][0],fn[i][j][1][1],fn[i][j][1][0] = rationalize(w_o,i_o,w_c,i_c)
								cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							if w not in p2s:						
								# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
								p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])
							else:
								if np.linalg.norm(cms_original-cms) > p2s[w][0]:
									# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
									p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])

							if w not in t2s:
								# t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							else:
								# if l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0])> t2s[w][0]:
								# 	t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								if l2_scores(i_o-i_c)> t2s[w][0]:
									t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							tot+=1

						except:
							# print("Skipped sentence #",i,j)
							cntim+=1
							pass
				print("Skipped",cntim,tot)
				lom_list.append(p2s)
				l2_list.append(t2s)
			except:
				print("Fail on sentence: %d"%idx)
				lom_list.append(None)
				l2_list.append(None)


		candidate_ppl = []

		for p,t,idx in zip(lom_list,l2_list,ids):
			if p==None or t==None or orig_interp==None or orig_interp[idx][1]==None:
				print("Skipped,",idx)
				continue

			for per_w in range(1,len(p)):
				try:
					# print(p[per_w][1][0])
					candidate_ppl.append(t[per_w][1][0])
				except:
					pass

			# i0,f0 = orig_interp[idx][1][0],orig_interp[idx][1][1]
			# sent_len = len(f0)-2

			# for per_w in range(1,len(p)):
			# 	try:
			# 		# results_lom.append(rank_correlation( i0, p[per_w][1][1][0])[0] )
			# 		# results_l2.append(rank_correlation( i0, t[per_w][1][1][0])[0] )
			# 		results_lom.append(topk_intersection( i0, p[per_w][1][1][0]) )
			# 		results_l2.append(topk_intersection( i0, t[per_w][1][1][0]) )
			# 		resultsx.append(per_w/sent_len)
			# 	except:
			# 		print("error")
			# 		pass
			print(idx)

		# for idx in range(args.number):
		# 	dataset_ppl.append(ta_dataset[idx][0][keyword])
		# 	for i in range(len(candidates[idx])):
		# 		for j in range(len(candidates[idx][i])):
		# 			candidate_ppl.append(candidates[idx][i][j].attacked_text.text.lower())
		# 	print(idx)

		# dataset_ppl_num = get_ppl(dataset_ppl)
		candidate_ppl_num = get_ppl(candidate_ppl)
		# print("Average dataset Perplexity:",dataset_ppl_num)
		print("Average Candidates Perplexity:",candidate_ppl_num)

	if args.metric == "grm":

		if args.dataset == "sst2":
			ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
			keyword = "sentence"
		elif args.dataset == "agnews":
			ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
			keyword = "text"
		elif args.dataset == "imdb":
			ta_dataset = HuggingFaceDataset("imdb", split=args.split)
			keyword = "text"

		candidate_name = str(args.candidatefolder)+"candidates"+"-"+str(args.dataset)+"-"+str(args.model)+".pkl"

		with open(candidate_name, 'rb') as f:
			candidates = pickle.load(f)

		if args.number == -1:
			args.number = len(candidates)

		dataset_grm = []
		candidate_grm = []

		for idx in range(args.number):
			dataset_grm.append(ta_dataset[idx][0][keyword])
			c = []
			for i in range(len(candidates[idx])):
				for j in range(len(candidates[idx][i])):
					# candidate_ppl.append(candidates[idx][i][j].attacked_text.text.lower())
					c.append(candidates[idx][i][j].attacked_text.text.lower())
			print(idx)
			candidate_grm.append(c)

	
		dataset_grm_num = get_grm_err(dataset_grm,candidate_grm)
		print("Average Grammar errors:",dataset_grm_num)

	if args.metric == "grm-selected":
		if args.dataset == "sst2":
			ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
			keyword = "sentence"
		elif args.dataset == "agnews":
			ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
			keyword = "text"
		elif args.dataset == "imdb":
			ta_dataset = HuggingFaceDataset("imdb", split=args.split)
			keyword = "text"

		interpretation_name = args.interpretfolder+"interpretations-"+args.dataset+"-"+args.model+"-"+args.interpretmethod+'-'+str(args.number)+".pkl"
		original_interpretation_name = args.interpretfolder+"original_sentences/original-interpretations-"+args.dataset+"-"+args.model+"-"+args.interpretmethod+'-'+str(args.number)+".pkl"
		
		with open(interpretation_name, 'rb') as f:
			interp = pickle.load(f)

		with open(original_interpretation_name, 'rb') as f:
			orig_interp = pickle.load(f)

		candidate_name = str(args.candidatefolder)+"candidates"+"-"+str(args.dataset)+"-"+str(args.model)+".pkl"

		with open(candidate_name, 'rb') as f:
			candidates = pickle.load(f)

		if args.number == -1:
			args.number = len(candidates)

		dataset_ppl = []
		candidate_ppl = []

		if args.number == -1:
			args.number = len(fin)

		lom_list = []
		l2_list = []

		ids = range(args.number)
		for fn,idx in zip(interp,ids):
			# if orig_interp[idx][1] == None:
			# 	print("Skippped due to external factors")
			# 	continue
			try:
			# print(orig_interp[idx])
				print("Calculating on %d of %d total sentences" %(idx,len(ids)))
				p2s = {}
				t2s = {}
				cms_original = com_scores(orig_interp[idx][1][0],orig_interp[idx][1][1])
				cntim = 0
				tot = 0
				for i in range(len(fn)):
					for j in range(len(fn[i])):
						try:
							w = len(diff(AttackedText(orig_interp[idx][0]),AttackedText(fn[i][j][0])))
							cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							w_o = orig_interp[idx][1][1]
							i_o = orig_interp[idx][1][0]
							w_c = fn[i][j][1][1]
							i_c = fn[i][j][1][0]

							if orig_interp[idx][1][0].size() != fn[i][j][1][0].size():
								
								w_o,i_o,w_c,i_c = rationalize(w_o,i_o,w_c,i_c)
								# orig_interp[idx][1][1],orig_interp[idx][1][0],fn[i][j][1][1],fn[i][j][1][0] = rationalize(w_o,i_o,w_c,i_c)
								cms = com_scores(fn[i][j][1][0],fn[i][j][1][1])

							if w not in p2s:						
								# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
								p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])
							else:
								if np.linalg.norm(cms_original-cms) > p2s[w][0]:
									# p2s[w] = (np.linalg.norm(cms_original-cms),fn[i][j])
									p2s[w] = (np.linalg.norm(cms_original-cms),[fn[i][j][0],[i_c,w_c]])

							if w not in t2s:
								# t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							else:
								# if l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0])> t2s[w][0]:
								# 	t2s[w] = (l2_scores(orig_interp[idx][1][0]-fn[i][j][1][0]),fn[i][j])
								if l2_scores(i_o-i_c)> t2s[w][0]:
									t2s[w] = (l2_scores(i_o-i_c),[fn[i][j][0],[i_c,w_c]])
							tot+=1

						except:
							# print("Skipped sentence #",i,j)
							cntim+=1
							pass
				print("Skipped",cntim,tot)
				lom_list.append(p2s)
				l2_list.append(t2s)
			except:
				print("Fail on sentence: %d"%idx)
				lom_list.append(None)
				l2_list.append(None)


		candidate_grm = []

		dataset_grm = []

		for p,t,idx in zip(lom_list,l2_list,ids):
			if p==None or t==None or orig_interp==None or orig_interp[idx][1]==None:
				print("Skipped,",idx)
				continue
			c = []
			dataset_grm.append(ta_dataset[idx][0][keyword])
			for per_w in range(1,len(p)):
				try:
					# print(p[per_w][1][0])
					c.append(t[per_w][1][0])
				except:
					pass
			candidate_grm.append(c)

			print(idx)

		print("Average Candidates Perplexity:",get_grm_err(dataset_grm,candidate_grm))

	if args.metric == "conf":

		if args.dataset == "sst2":
			ta_dataset = HuggingFaceDataset("glue", "sst2", args.split)
			keyword = "sentence"
		elif args.dataset == "agnews":
			ta_dataset = HuggingFaceDataset("ag_news", "test",split=args.split)
			keyword = "text"
		elif args.dataset == "imdb":
			ta_dataset = HuggingFaceDataset("imdb", split=args.split)
			keyword = "text"

		candidate_name = str(args.candidatefolder)+"candidates"+"-"+str(args.dataset)+"-"+str(args.model)+".pkl"

		with open(candidate_name, 'rb') as f:
			candidates = pickle.load(f)

		if args.number == -1:
			args.number = len(candidates)

		candidate_conf = []
		model = model.to(device)

		cfws = {}
		gt=[]

		for idx in range(args.number):
			for i in range(len(candidates[idx])):
				for j in range(len(candidates[idx][i])):
					# print(torch.argmax(candidates[idx][i][j].raw_output).item(),ta_dataset[idx][1])
					if torch.argmax(candidates[idx][i][j].raw_output).item() == ta_dataset[idx][1]:
						# inp = torch.tensor(tokenizer.encode(ta_dataset[idx][0][keyword],truncation=True)).to(device)
						# inp = inp.unsqueeze(0)
						# o = model(inp)
						# gt.append(torch.max(Softmax()(o[0])).item()) 
						w = len(diff(AttackedText(ta_dataset[idx][0][keyword]),candidates[idx][i][j].attacked_text))
						ii = candidates[idx][i][j].raw_output
						if w not in cfws:
							cfws[w] = [torch.max(ii).item()]
						else:
							cfws[w].append(torch.max(ii).item())
						# print(ii)
						candidate_conf.append(torch.max(ii).item())
					# exit(0)
			print(idx)

		print("Average Model confidence:",np.mean(candidate_conf))
		for i in cfws:
			cfws[i] = np.mean(cfws[i])
		print("Perturbed Model confidence::")
		print("Perturbed:0",np.round(np.mean(gt),2))
		for i in cfws:
			print("Perturbed:",i," ",np.round(cfws[i],2))


main()
			

