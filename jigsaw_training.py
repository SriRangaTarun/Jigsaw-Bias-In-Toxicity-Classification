# Import necessary packages

import gc
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from custom_layers import AttentionWeightedAverage, Capsule, squash

# Define hyperparameters and paths

parser = argparse.ArgumentParser()
parser.add_argument('embedding_path')
parser.add_argument('train_data_path')

args = parser.parse_args()
embding_path = args.embedding_path
train_data_path = args.train_data_path

MAXLEN = 128
NUM_EPOCHS = 100
EMBED_SIZE = 300
BATCH_SIZE = 2048
MAX_FEATURES = 100000

# Load data and define tokenizer

train_df = pd.read_csv(train_data_path)
tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True)

# Fit tokenizer to the comments

tokenizer.fit_on_texts(list(train_df['comment_text']))
X_train = tokenizer.texts_to_sequences(list(train_df['comment_text']))
X_train = pad_sequences(X_train, maxlen=MAXLEN); del tokenizer; gc.collect()

# Build model using custom Capsule and Attention layers
# 1). Takes embedded sentences as input
# 2). Passes the output through spatial dropout and a Bidirectional LSTM layer
# 3). The LSTM output is passed through Max Pooling and Weighted Average Attention
# 4). The same LSTM output also passed through a Capsule layer and a Flatten layer
# 5). The outputs of Pooling, Attention and Capsule are then concatenated into a vector
# 6). They are then passed through a dense layer with one neuron followed by a Sigmoid activation

# We use binary crossentropy as the loss function and ADAM as the optimizer

def get_model():
    inp = Input(shape=(MAXLEN,))
    embed = Embedding(3303 + 1, EMBED_SIZE, input_length=MAXLEN, trainable=False)
    
    embed_inp = embed(inp)
    drop_inp = SpatialDropout1D(0.3)(embed_inp)
    bi_lstm = Bidirectional(CuDNNLSTM(64, return_sequences=True))(drop_inp)
    
    max_pool_lstm = GlobalMaxPooling1D()(bi_lstm)
    attention_lstm = AttentionWeightedAverage()(bi_lstm)
    capsule = Flatten()(Capsule(num_capsule=5, dim_capsule=5, routings=4, activation=squash)(bi_lstm))
    outp = Dense(1, activation='sigmoid')(concatenate([max_pool_lstm, attention_lstm, capsule], axis=1))
    
    model = Model(inp, outp)
    loss = 'binary_crossentropy'
    optimizer = Adam(lr=0.005, decay=0.001)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc']); return model

# Train model on five folds (with callbacks) and save model after each fold

for fold in [0, 1, 2, 3, 4]:
    K.clear_session()
    tr_ind, val_ind = splits[fold]
     
    model = get_model()
    ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    model.fit(X_train[tr_ind],
              y_train[tr_ind]>0.5,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(X_train[val_ind], y_train[val_ind]>0.5),
              callbacks = [es, ckpt])

# Save word index

word_index = tokenizer.word_index
with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
