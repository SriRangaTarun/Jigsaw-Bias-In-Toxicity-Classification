import gc
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from keras.layers import *
from keras.optimizers import *
from keras.activations import *
from keras.constraints import *
from keras.initializers import *
from keras.regularizers import *

import keras.backend as K
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('embedding_path')
parser.add_argument('test_data_path')
parser.add_argument('train_data_path')

args = parser.parse_args()
embding_path = args.embedding_path
test_data_path = args.test_data_path
train_data_path = args.train_data_path


MAXLEN = 128
NUM_EPOCHS = 100
EMBED_SIZE = 300
BATCH_SIZE = 2048
MAX_FEATURES = 100000

test_df = pd.read_csv(test_data_path)
train_df = pd.read_csv(train_data_path)
