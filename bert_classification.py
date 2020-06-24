from transformers import BertTokenizer,BertModel
import torch
import json 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score



language_A = 'id'
language_B = 'fr'
language_C = 'es'


def reduce_dims(embeddings, reduction="PCA"):
  
    if reduction == "TSNE":
        X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    elif reduction == "PCA":
        X_embedded = PCA(n_components=2).fit_transform(embeddings)

    return X_embedded

def preprocess_data(file):

	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') 
	model = BertModel.from_pretrained('bert-base-multilingual-cased') 

	with open(file,"r",encoding="latin-1") as json_file:
		labeled_data = json.load(json_file)

	data = []

	for k in tqdm(list(labeled_data.keys())):
		orig_sentence = k

		if labeled_data[k]['best'] == 'ac':
			label = 0
		elif labeled_data[k]['best'] == 'abc':
			label = 1
		else:
			continue

		# if labeled_data[k]['best'] == 'ac':

		# print(labeled_data[k].keys())

		# if labeled_data[k]['er']-labeled_data[k]['fr']  > 0.05:
		# 	label = 1
		# else:
		# 	label = 0


		all_out,embed = model(torch.tensor([tokenizer.encode(orig_sentence)]))
		
		data.append((orig_sentence,np.append(torch.mean(all_out,1).view(-1).data.numpy(),len(orig_sentence.split())),label)) 

	return data


def classify(embeddings,labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.20, random_state=1)
       
    model = LogisticRegression().fit(X_train, y_train)

    # model = DecisionTreeClassifier().fit(X_train, y_train)
    # model = DecisionTreeClassifier().fit(X_train, y_train)
    # model = XGBClassifier().fit(X_train, y_train)

    print("Score:",model.score(X_test, y_test))

    disp = plot_precision_recall_curve(model, X_test, y_test)
    y_score = model.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)

    print("AP:",average_precision)
    disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

if __name__ == "__main__":

	
	data = preprocess_data("label_data_{}_{}_{}.json".format(language_A,language_B,language_C))
	embeddings = [d[1] for d in data]
	labels = [d[2] for d in data]
	embeddings = reduce_dims(embeddings)
	classify(embeddings,labels)














