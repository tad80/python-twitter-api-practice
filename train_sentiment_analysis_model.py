import os
import sys
import json
import re
import shutil
import logging
from pathlib import Path
import platform

import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from tqdm.auto import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # Registers the ops.
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from official.nlp import optimization

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.bigquery import magics

nltk.download('punkt')

# Suppress Info and Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disables AVX2 FMA warnings (CPU support)


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
print('Server Platform:', platform.platform())
print()
print('GPU Information:')

BATCH_SIZE = 64
MODEL_PATH= "{model}_sst2.h5"
FINAL_MODEL_PATH = 'finetuned_model'
LEARNING_RATE = 1e-5
EPOCHS = 20           
PREPROCESSOR_NAME = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
MODEL_NAME = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2'
PRECLASSIFIER_DIMS = int(int(re.search("_H-(\d{3})_", MODEL_NAME)[1])/2)
DROPOUT = 0.2

# Can set this to None to have it auto-calculated.
NUM_WARMUP_STEPS = 600

SAVE_MODEL_PATH = MODEL_PATH.format(model=re.sub(r"[^\w\-]+", 
                                                 '_', 
                                                 MODEL_NAME.replace('https://tfhub.dev/tensorflow', ''))
                                    )

# Modified From: https://github.com/ForeverZyh/certified_lstms/blob/81ca8c66c6e0d1a15f8abcd513e9370bd95dfb8b/src/text_classification.py
def prepare_ds(ds):
    text_list = []
    label_list = []
    num_pos = 0
    num_neg = 0
    num_words = 0
    for features in tqdm(tfds.as_numpy(ds), total=len(ds)):
        sentence, label = features["sentence"], features["label"]
        tokens = word_tokenize(sentence.decode('UTF-8').lower())
        text_list.append(' '.join(tokens))
        label_list.append(label)
        num_pos += label == 1
        num_neg += label == 0
        num_words += len(tokens)

    avg_words = num_words / len(text_list)
    print('Read %d examples (+%d, -%d), average length %d words' % (
        len(text_list), num_pos, num_neg, avg_words))
    return tf.data.Dataset.from_tensor_slices((text_list, label_list))


# TFDS Glue URLs are currently broken
tfds.text.glue.Glue.builder_configs['sst2'].data_url = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'

# Load from Tensorflow Datasets
train, valid = tfds.load(
    name="glue/sst2",
    with_info=False,
    split=['train', 'validation']
)


print()
print('Building Training Data')
train_dataset = prepare_ds(train).shuffle(1000).batch(BATCH_SIZE)
print()
print('Building Validation Data')
valid_dataset = prepare_ds(valid).batch(BATCH_SIZE)
print()


def build_classifier_model(train_dataset):
  text_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(PREPROCESSOR_NAME, name='preprocessing', )
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(MODEL_NAME, trainable=True, name='BERT_encoder', )
  outputs = encoder(encoder_inputs)

  net = outputs['pooled_output']
  net = tf.keras.layers.Dense(
              int(PRECLASSIFIER_DIMS),
              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.002),
              activation="relu",
              name="pre_classifier"
          )(net)

  net = tf.keras.layers.Dropout(DROPOUT)(net)
  net = tf.keras.layers.Dense(2, activation="sigmoid", use_bias=True, name='classifier')(net)
  model = tf.keras.Model(text_input, net, name='sentiment_classification')

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
  num_train_steps = steps_per_epoch * EPOCHS
  num_warmup_steps = NUM_WARMUP_STEPS or int(0.1*num_train_steps)

  optimizer = optimization.create_optimizer(init_lr=LEARNING_RATE,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

  model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy'])

  model.summary()
  return model


model = build_classifier_model(train_dataset)

callbacks = [ModelCheckpoint(filepath=SAVE_MODEL_PATH, 
                             verbose=1,
                             save_freq='epoch',
                             monitor='val_accuracy',
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only=True),
             EarlyStopping(patience=3, monitor='val_loss', mode='min')
            ]


# Training model...
history = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_data=valid_dataset)

model = build_classifier_model(train_dataset)
model.load_weights(SAVE_MODEL_PATH)

tf.saved_model.save(
    model,
    FINAL_MODEL_PATH,
    signatures=None,
    options=None, 
)


def get_size(path: str) -> int:
    def filesize(size: int) -> str:
        for unit in ("B", "K", "M", "G"):
            if size < 1024:
                break
            size /= 1024
        return f"{size:.1f}{unit}"
    return filesize(sum(p.stat().st_size for p in Path(path).rglob('*')))

print()
print('Model Size:')
print(get_size(FINAL_MODEL_PATH))


loaded = tf.saved_model.load(FINAL_MODEL_PATH)
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(infer.structured_input_signature)
print(infer.structured_outputs)


example_texts =  [ "3m products are really good",
                  "when did 3m move to china",
                  "is 3m a good brand",
                  "are 3m products made in usa",
                  "does 3m make you sick",
                  "is 3m canadian or american",
                  "what does 3m stand for",
                  "how many precious metals are there",
                  "are kmart stores closing",
                  "what are the 5 most precious metals",
                  "what are the 4 precious metals",
                  "is 3m illegal",
                  "is 3m legit",
                  "is 3m legal",
                  "what is the best precious metal to buy right now",
                  "what are the 4 most precious metals",
                  "how can you tell if a precious metal is real",
                  "what are the four precious metals",
                  "what is the cheapest precious metal",
                  "what is abbott laboratories known for",
                  "is abbott a bad company"
                ]

labels = {0: 'NEGATIVE',
          1: 'POSITIVE'}

input_data = tf.convert_to_tensor(example_texts, dtype=tf.string)

output = loaded(input_data)

[{"label": labels[item.numpy().argmax()], "score": item.numpy().max(), "paa": example_texts[i]} for i, item in enumerate(output)]


