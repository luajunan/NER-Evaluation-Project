# NER Evaluation Framework

## Project Description

The main goal for this project is to evaluate and find out how current existing Named-entity Recognition (NER) models fare across different languages. The framework aims to evaluate the F1, Recall and Precision scores across various Named-entity Recognition (NER) models. These models include Polyglot, SpaCy, NLTK, Deeppavlov, Flair and XLM-Roberta.

## Getting Started
If you are running your code on google colab, the installation is as follows

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
model = tner.TransformersNER("asahi417/tner-xlm-roberta-large-ontonotes5")```
