
from transformers import MarianMTModel, MarianTokenizer
import torch
import nltk
from tqdm import tqdm
import numpy as np
import json


#Schema: A->B->C, A->C
#A: French (Fr)
#B: Spanish (Es)
#C: Romanian (RO)

language_A = 'id'#'fr'
language_B = 'fr'#'es'
language_C = 'es' #ro'
DATA_FILE = language_A+"-"+language_C+"/opus-2020-01-16.test.txt" #"fr-ro/opus-2020-01-16.test.txt"
SAVE = True

def process_paired_data(file,n=None):

	a_input = []	
	c_gold = []
	with open(file,"r",encoding="latin-1") as paired_file:
		data = paired_file.read().split("\n\n")
		if n is not None:
			data = data[:n]
		for l in data:
			sentences = l.split("\n")
			if len(sentences) == 3:
				a_input.append(sentences[0])
				c_gold.append(sentences[1])

	return a_input,c_gold

def translate(sentences,inp_lang,out_lang):
	tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-{}-{}".format(inp_lang,out_lang))
	model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-{}-{}".format(inp_lang,out_lang))

	output = []
	for sentence in tqdm(sentences):
		translated = model.generate(**tokenizer.prepare_translation_batch([sentence]))
		output.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])

	return output

def compare_performance(a_input,c_from_a,c_from_b,c_gold):
	ac_best = []
	abc_best = []

	labeled_data = {}

	for orig,ac,abc,gold in zip(a_input,c_from_a,c_from_b,c_gold):
		
		BLEUscore_ac = nltk.translate.bleu_score.sentence_bleu([ac], gold)
		BLEUscore_abc = nltk.translate.bleu_score.sentence_bleu([abc], gold)

		print(BLEUscore_ac,BLEUscore_abc)

		if BLEUscore_ac > BLEUscore_abc+0.01:
			ac_best.append((gold,BLEUscore_ac,BLEUscore_ac-BLEUscore_abc))
			best = "ac"
		elif  BLEUscore_ac < BLEUscore_abc-0.01:
			abc_best.append((gold,BLEUscore_abc,BLEUscore_abc-BLEUscore_ac))
			best = "abc"
		else:
			best = "EQUAL"

		labeled_data[orig]={"best":best,"fr":BLEUscore_ac,"er":BLEUscore_abc}

	return ac_best,abc_best,labeled_data


if __name__ == "__main__":

	a_input,c_gold = process_paired_data(DATA_FILE,n=1000)
	b_from_a = translate(a_input,language_A,language_B)
	c_from_b = translate(b_from_a,language_B,language_C)
	c_from_a = translate(a_input,language_A,language_C)

	ac_best,abc_best,labeled_data = compare_performance(a_input,c_from_a,c_from_b,c_gold)

	if SAVE:
		with open("label_data_{}_{}_{}.json".format(language_A,language_B,language_C),"w+",encoding="latin-1") as json_file:
			json.dump(labeled_data,json_file)

	print("A->C Wins: ",len(ac_best),ac_best)
	print("\n\n\n\n")
	print("A->B->C Wins: ",len(abc_best),abc_best)



#TODO
#For classifier, look at sentence side, number of negations, stop vords, punctionation
#Write script that finds best bleu pair!
#Find all paths of length 2, such that A->B-C, A->C Are possible, rank by difference between average A->B->C




