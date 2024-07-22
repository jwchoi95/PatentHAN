import os
from transformers.models.bert.modeling_bert import BertModel
import pickle
from tqdm import tqdm
from nltk import sent_tokenize
import string
import re
import pandas as pd
import numpy as np
import random as r
import torch
from transformers import *
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

r.seed(2019)

np.random.seed(2019)


os.environ['PYTHONHASHSEED'] = str(2019)

args = {
    'cuda_num': 0,

    # directory for original text data
    'text_data_dir': 'raw_corpora/',


    'data_dir': 'datasets/',

}


def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))



def clean_corpus(corpus):

    cleaned_corpus = []

    for article in corpus:

        article = article.lower()
        temp_str = re.sub(r'\d+', '', article)
        temp_str = re.sub(r'[^\x00-\x7f]', r'', temp_str)
        temp_str = temp_str.translate(
            str.maketrans('', '', string.punctuation))
        temp_str = re.sub(r'\s+', ' ', temp_str)

        cleaned_corpus.append(temp_str)

    return cleaned_corpus


def read_data(path):

    corpora = []
    for filename in os.listdir(path):

        df_temp = pd.read_csv(path+filename, encoding='utf8')  # iso-8859-1

        corpora.append(df_temp.text.tolist())

    class_one_len = len(corpora[0])
    class_two_len = len(corpora[1])

    return corpora, class_one_len, class_two_len


# split a document into sentences
def sentences_segmentation(corpora, tokenizer, min_token=0):
    segmented_documents = []

    for document in tqdm(corpora):

        segmented_document = []
        seg_document = sent_tokenize(document)


        for sentence in seg_document:
            tokenized_sentence = tokenizer.tokenize(sentence)
            if len(tokenized_sentence) > min_token:
                temp_sentence = tokenizer.convert_tokens_to_string(
                    tokenized_sentence)

                if not all([j.isdigit() or j in string.punctuation for j in temp_sentence]):
                    segmented_document.append(temp_sentence)

        segmented_documents.append(segmented_document)

    return segmented_documents




def encode_bert_hbm(model_name, dataset_name):

    if model_name == 'patBERT':

        tokenizer = AutoTokenizer.from_pretrained(
            "prithivida/bert-for-patents-64d")
        model = AutoModel.from_pretrained("prithivida/bert-for-patents-64d")

    elif model_name == 'sciBERT':
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
        tokenizer = AutoTokenizer.from_pretrained(
            'allenai/scibert_scivocab_cased')

    elif model_name == 'roBERTa':
        model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    elif model_name == 'BERT':
        model = BertModel.from_pretrained("bert-base-cased")
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case=False)

    elif model_name == 'BART':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartModel.from_pretrained('facebook/bart-base')

    elif model_name == 'bioBERT':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
    
    elif model_name == 'pubBERT':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")


    model.eval()
    model.cuda(args['cuda_num'])

    segmented_documents = pickle.load(open('segmented_documents_26.pkl', 'rb'))


    doc_sen_embeddings = []

    segmented_documents = segmented_documents[30000:]
    for doc in tqdm(segmented_documents):


        doc_sen_embedding = []
        for sen in tqdm(doc):
            input_ids = tokenizer(sen)['input_ids']

       
            if len(input_ids) > 512:
                input_ids = input_ids[:512]

            tokens_tensor = torch.tensor([input_ids]).cuda(args['cuda_num'])
            encoded_layers = model(tokens_tensor)

            embeddings_array = encoded_layers[0][0].cpu().detach().numpy()

            del encoded_layers
            del tokens_tensor

            doc_sen_embedding.append(embeddings_array)

        doc_sen_embeddings.append(doc_sen_embedding)

    
    doc_sen_avg_embeddings = []
    for doc in doc_sen_embeddings:

        
        temp_doc = []
        for sen in doc:
            avg_sen = np.mean(sen, axis=0)
            temp_doc.append(avg_sen)
        doc_sen_avg_embeddings.append(np.array(temp_doc))

    doc_sen_avg_embeddings = np.array(doc_sen_avg_embeddings)

    doc_dict = {}
    for i in range(len(doc_sen_avg_embeddings)):
        doc_dict[i] = doc_sen_avg_embeddings[i]
    pickle.dump(doc_dict, open(os.path.join(
        args['data_dir'], 'patent_26-base_data/%s_%s.p' % (value, dataset_name)), 'wb'))


if __name__ == "__main__":

    dataset_name = 'dataset'
    
    encoding_methods = ['patBERT','sciBERT']

    for value in encoding_methods:

        if value in ['patBERT', 'sciBERT', 'roBERTa', 'BERT', 'BART', 'bioBERT','pubBERT']:
            print('starting coding hbm')
            encode_bert_hbm(value, dataset_name)
