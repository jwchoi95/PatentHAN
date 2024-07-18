<h1 align="left">PatentHAN</h1>
<h2 align="left">Implementation of "Early screening of potential breakthrough technologies with enhanced interpretability: A patent-specific hierarchical attention network model"</h3>


### Dataset


### Requirements
Our experiment setting is as follows:


### Install
```bash
pip install -r requirements.txt
```


### PLM configuration
```python
from transformers import *

#PatentBERT
tokenizer = AutoTokenizer.from_pretrained("dheerajpai/patentbert")
model = AutoModelForMaskedLM.from_pretrained("dheerajpai/patentbert")

#BERT from arXiv preprint arXiv:1810.04805.
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

#SciBERT from arXiv preprint arXiv:1903.10676.
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

#BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
model = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

#PubMedBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")


