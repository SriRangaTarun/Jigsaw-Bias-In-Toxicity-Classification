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
train_df = pd.read_csv(train_data_path)
tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True)

tokenizer.fit_on_texts(list(train_df['comment_text']))
X_train = tokenizer.texts_to_sequences(list(train_df['comment_text']))
X_train = pad_sequences(X_train, maxlen=MAXLEN); del tokenizer; gc.collect()

word_index = tokenizer.word_index

def get_model():
    inp = Input(shape=(MAXLEN,))
    embed = Embedding(3303 + 1, EMBED_SIZE, input_length=MAXLEN, trainable=False)
    
    embed_inp = embed(inp)
    drop_inp = SpatialDropout1D(0.3)(embed_inp)
    bi_lstm = Bidirectional(CuDNNLSTM(64, return_sequences=True))(drop_inp)
    
    max_pool_lstm = GlobalMaxPooling1D()(bi_lstm)
    attention_lstm = AttentionWeightedAverage()(bi_lstm)
    capsule = Capsule(num_capsule=5, dim_capsule=5, routings=4, activation=squash)(bi_lstm)
    
    x = [max_pool_lstm, attention_lstm, Flatten()(capsule)]
    outp = Dense(1, activation='sigmoid')(concatenate(x, axis=1))
    
    model = Model(inp, outp)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005, decay=0.001), metrics=['acc'])
    
    return model

model = get_model()
split = np.int32(0.8*len(X_train))
ckpt = ModelCheckpoint(f'model.h5', save_best_only = True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

model.fit(X_train[:split],
          y_train[:split]>0.5,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS//2,
          callbacks = [es, ckpt],
          validation_data=(X_train[split:], y_train[split:]>0.5))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
