import urllib.request
import urllib.error
from lxml import etree
import numpy as np
import requests
import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
import json
import random
from tqdm import tqdm
import unicodedata


def get_model_list(filter_val):

    filtered_results = []

    hugging_face_models = "https://huggingface.co/models" 
    page = urllib.request.urlopen(hugging_face_models)
    soup = BeautifulSoup(page, 'lxml')

    data = soup.findAll('ul',attrs={'class':'models-list'})
    for el in data:
        links = el.findAll('a')
        for a in links:
            if filter_val in a.text:
                result = "https://huggingface.co" + a['href']
                print(result)

                filtered_results.append(result)
    return filtered_results

def extract_model_info(mt_url):

    languages = mt_url.split('-')[-2:]

    page = urllib.request.urlopen(mt_url)
    soup = BeautifulSoup(page, 'lxml')

    data = soup.findAll('table')
    for el in data:
        if 'BLEU' in el.text:
            benchmark = el.findAll('tr')[-1].text.split()

            name = benchmark[0]
            bleu = float(benchmark[1])
            chr_f = float(benchmark[2])

            return (languages[0],languages[1],name,bleu,chr_f)

    return (languages[0],languages[1],None,None,None)
        
def find_best_path(pair,score_dict,aggregation_func=lambda x: np.mean(x)):
	start = pair.split('-')[0]
	end = pair.split('-')[1]

	all_possible_start = set([k.split('-')[1]  for k in score_dict.keys() if start == k.split('-')[0] and score_dict[k] is not None])
	all_possible_end = set([k.split('-')[0] for k in score_dict.keys() if end == k.split('-')[1] and score_dict[k] is not None])

	candidates = all_possible_start.intersection(all_possible_end)

	candidate_aggre_bleu = [(b,aggregation_func([score_dict["{}-{}".format(start,b)],score_dict["{}-{}".format(b,end)]])) for b in  candidates]

	if len(candidate_aggre_bleu) == 0: return (None,None)

	return max(candidate_aggre_bleu,key=lambda x: x[1])

if __name__ == "__main__":

    results = get_model_list("Helsinki-NLP/opus-mt")[:80]

    language_pairs = {}

    for r in results:
        lang_A,lang_B, _ , bleu, _ = extract_model_info(r)
        pair = "{}-{}".format(lang_A,lang_B)
        language_pairs[pair] = bleu
        print(pair,bleu)

    for pair in language_pairs.keys():
    	if language_pairs[pair] == None: continue
    	print("Original:",(pair,language_pairs[pair]), "Other:",find_best_path(pair,language_pairs))
