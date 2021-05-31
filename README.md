# NER Evaluation Framework

## Project Description

The main goal for this project is to evaluate and find out how current existing Named-entity Recognition (NER) models fare across different languages. The framework aims to evaluate the F1, Recall and Precision scores across various Named-entity Recognition (NER) models. These models include Polyglot, SpaCy, NLTK, Deeppavlov, Flair and XLM-Roberta. The current framework only accepts 6 languages, namely English, Chinese (Simplified), Bahasa Melayu, Bahasa Indonesian, Vietnamese and Thai.

## Data Standardisation

| Word | Entity |
|----|----|
| Tim | B-per | 
| is | O |

Ensure that the coNLL file is of the following format before the framework can process it as an acceptable input.

## Standard Entity Set
[BIO tagging](https://medium.com/analytics-vidhya/bio-tagged-text-to-original-text-99b05da6664)

Dion's 0 shot learning model
* 'B-loc' - Beginning of location
* 'B-per' - Beginning of person
* 'B-org' - Beginning of organisation
* 'I-loc' - Inside of location
* 'I-per' - Inside of person
* 'I-org' - Inside of organisation
* 'O' - Outside

## Getting Started

### Installation of Packages
If you are running your code on google colab, the installation is as follows.
(Doing this on google colab might require you to restart the runtime several times due to the overlapping installation of certain packages)
```python
!pip install pypi
!pip install polyglot
!pip install PyICU
!pip install pycld2
!pip install Morfessor
!polyglot download LANG:zh
!polyglot download LANG:en
!polyglot download LANG:ms
!polyglot download LANG:id
!polyglot download LANG:vi
!polyglot download LANG:th
!polyglot download embeddings2.en
!polyglot download embeddings2.id
!polyglot download embeddings2.ms
!polyglot download embeddings2.vi
!polyglot download embeddings2.th

!pip install -U spacy
import spacy
!python -m spacy download zh_core_web_sm
!python -m spacy download en

!pip install nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags
nltk.download("maxent_ne_chunker")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("words")

!pip install deeppavlov
import deeppavlov

!python -m deeppavlov install squad_bert
from deeppavlov import configs, build_model

!pip install flair
from segtok.segmenter import split_single
from flair.data import Sentence
from flair.models import SequenceTagger

!pip install tner
import tner
model = tner.TransformersNER("asahi417/tner-xlm-roberta-large-ontonotes5")
```

### Importing Necessary Modules

```python
import polyglot as poly
from polyglot.text import Text, Word

import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags

import deeppavlov
from deeppavlov import configs, build_model

from segtok.segmenter import split_single
from flair.data import Sentence
from flair.models import SequenceTagger

import tner

import re
```
### Functions and Inputs

#### Preprocessing and Evaluation
```python
read_conll_file(filename)
read_file(filename)
clean_para(txt)
para_to_conll(text)
tabulate_score(actual_dic, model_dic, model_ent, model_name)
```

```read_conll_file(filename)``` takes in the filepath as the only argument. It returns a list with the paragraph of text as the first element and a list containing a list of word and tags as the second argument. eg: 
```python 
para, word_tag_list = read_conll_file('/data/sample_data.txt', 2)
```

```read_file(filename)``` takes in the filepath as the only argument and returns a paragraph of text. eg: 
```python 
para = read_file('/data/sample_data.txt')
```

```clean_para(txt)``` takes in a paragraph of text with embedded taggings within and returns a cleaned paragraph of text. eg: 
```python 
cleaned_para = clean_para(text)
```

```para_to_conll(text)``` takes in a paragraph of text and returns a dictionary with keys as NER tags and values as a list of entities that falls within the category. eg: 
```python 
model_dic = para_to_conll(text)
```

```tabulate_score(actual_dic, model_dic, model_ent, model_name)``` takes in the actual dictionary of the test set, the dictionary predicted by the model, the list of entities recognised by the model and the name of the model. It returns a list which contains the F1, Recall and Precision scores respectively. eg: 
```python
f1_score, recall_score, precision_score, model_name = tabulate_score(actual_dic, model_dic_polyglot, model_ent_polyglot, "Polyglot")
```

#### Models
```python
polyglot(text, language_code)
spa(text, language_code)
extract_ne_nltk(text)
extract_ne_deeppavlov(text, space=True)
extract_ne_flair(text)
extract_ne_roberta(text, space=True)
```

All the functions take in a standardized input of a paragraph of text as the first argument. The language_code applies only to the Polyglot and SpaCy models. The argument space is to indicate the presence of spaces between characters/words as some languages like Chinese do not have spaces between each character. All the outputs are standardized to return a list of two elements. The dictionary predicted by the model is the first element and the list of entities recognised by the model is the second element. eg: 
```python 
model_dic_poly, model_ent_poly = polyglot(text, 'en')
```

#### Main
```python
main(filename, language_code, file_type)
```

The main method consolidates all the functions and allows the user to use the framework more conveniently. It takes in the filename as the first input, the language_code, file-type (eg: "conll" or "text"). This method returns a list of lists where each list contains the F1, Recall, Precision and the name of the model and the outer list contains all the list of scores and models that were evaluated for the dataset. eg: 
```python 
results_list = main('/data/sample_data.txt', 'en', 'conll')
```
