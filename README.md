# NER Evaluation Framework

## Project Description

The main goal for this project is to evaluate and find out how current existing Named-entity Recognition (NER) models fare across different languages. The framework aims to evaluate the F1, Recall and Precision scores across various Named-entity Recognition (NER) models. These models include Polyglot, SpaCy, NLTK, Deeppavlov, Flair and XLM-Roberta.

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
!polyglot download embeddings2.en
!polyglot download pos2.en
!polyglot download embeddings2.id
!polyglot download pos2.id
!polyglot download embeddings2.ms
!polyglot download LANG:vi
!polyglot download embeddings2.vi
!polyglot download LANG:th
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
### Functions And Inputs

#### Preprocessing and Evaluation
```python
read_conll_file(filename, tag_col)
read_file(filename)
clean_para(txt)
para_to_conll(text)
tabulate_score(actual_dic, model_dic, model_ent, model_name)
```

```python read_conll_file(filename, tag_col)``` takes in the filepath and the column number of the NER tags (0-based indexing). It returns a list with the paragraph of text as the first element and a list containing a list of word and tags as the second argument.

```python read_file(filename)``` takes in the filepath as the only argument and returns a paragraph of text.

```python clean_para(txt)``` takes in a paragraph of text with embedded taggings within and returns a cleaned paragraph of text.

```python para_to_conll(text)``` takes in a paragraph of text and returns a dictionary with keys as NER tags and values as a list of entities that falls within the category.

```tabulate_score(actual_dic, model_dic, model_ent, model_name)``` takes in the actual dictionary of the test set, the dictionary predicted by the model, the list of entities recognised by the model and the name of the model. It returns a list which contains the F1, Recall and Precision scores respectively.

#### Models
```python
polyglot(text, language_code)
spa(text, language_code)
extract_ne_nltk(text)
extract_ne_deeppavlov(text, space=True)
extract_ne_flair(text)
extract_ne_roberta(text, space=True)
```

All the functions take in a standardized input of a paragraph of text as the first argument. The language_code applies only to the Polyglot and SpaCy models. The argument space is to indicate the presence of spaces between characters/words as some languages like Chinese do not have spaces between each character. All the outputs are standardized to return a list of two elements. The dictionary predicted by the model is the first element and the list of entities recognised by the model is the second element.
