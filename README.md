# Sequence-Classification-with-Transformers
Bert 

This colab notebook will guide you through using the Transformers library to obtain state-of-the-art results on the sequence classification task. It is attached to the following tutorial.

We will be using two different models as a means of comparison: Google's BERT and Facebook's RoBERTa. Both have the same architecture but have had different pre-training approached.

import the TensorFlow modules, transformers

import tensorflow as tf

!pip install transformers

from transformers import (TFBertForSequenceClassification, 
                          BertTokenizer,
                          TFRobertaForSequenceClassification, 
                          RobertaTokenizer)

bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
