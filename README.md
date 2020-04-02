# Sequence-Classification-with-Transformers (Bert)

This article will guide you through using the Transformers library to obtain state-of-the-art results on the sequence classification task. It is attached to the following tutorial. We will be using Bert model as a means of comparison: Google's BERT.

Import the TensorFlow modules:
               
    import tensorflow as tf

Import the transformers:

    !pip install transformers

    from transformers import (TFBertForSequenceClassification, BertTokenizer)

    bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

Getting the dataset:

    import tensorflow_datasets

    data = tensorflow_datasets.load("glue/mrpc")

    train_dataset = data["train"]

    validation_dataset = data["validation"]

Encoding sequences:

    encoded_bert_sequence = bert_tokenizer.encode(seq0, seq1, add_special_tokens=True, max_length=128)

    bert_special_tokens = [bert_tokenizer.sep_token_id, bert_tokenizer.cls_token_id]

Encode_plus():

    from transformers import glue_convert_examples_to_features

    bert_train_dataset = glue_convert_examples_to_features(train_dataset, bert_tokenizer, 128, 'mrpc')
    
    bert_train_dataset = bert_train_dataset.shuffle(100).batch(32).repeat(2)

    bert_validation_dataset = glue_convert_examples_to_features(validation_dataset, bert_tokenizer, 128, 'mrpc')
    
    bert_validation_dataset = bert_validation_dataset.batch(64)

The two BERT datasets are now ready to be used: the training dataset is shuffled and batch, while the validation dataset is only batched.

....................................Training the model....................................

Before fine-tuning the model, we must define a few hyperparameters that will be used during the training such as the optimizer, the loss and the evaluation metric.

As an optimizer we'll be using Adam, which was the optimizer used during those models' pre-training. As a loss we'll be using the sparse categorical cross-entropy, and the sparse categorical accuracy as the evaluation metric.

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

The beauty of tensorflow/keras lies here: using keras' fit method to fine-tune the model with a single line of code 

        bert_history = bert_model.fit(bert_train_dataset, epochs=3, validation_data=bert_validation_dataset)

Evaluating the BERT model:

    bert_model.evaluate(bert_validation_dataset)











