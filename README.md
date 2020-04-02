# Sequence-Classification-with-Transformers
Bert 

This colab notebook will guide you through using the Transformers library to obtain state-of-the-art results on the sequence classification task. It is attached to the following tutorial.

We will be using two different models as a means of comparison: Google's BERT and Facebook's RoBERTa. Both have the same architecture but have had different pre-training approached.

#import the TensorFlow modules

import tensorflow as tf

#import the transformers

!pip install transformers

from transformers import (TFBertForSequenceClassification, BertTokenizer)

bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#Getting the dataset

import tensorflow_datasets

data = tensorflow_datasets.load("glue/mrpc")

train_dataset = data["train"]

validation_dataset = data["validation"]

#textBERT Tokenizer


