import numpy as np

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(embed_dir=EMB_PATH):
    return dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir))

def build_embedding_matrix(word_index, embeddings_index, MAX_FEATURES, lower=True):
    size = (MAX_FEATURES, EMBED_SIZE)
    embedding_matrix = np.zeros(size)

    for word, i in word_index.items():
        if lower: word = word.lower()
        if i >= MAX_FEATURES: continue
        try: embedding_vector = embeddings_index[word]
        except: embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def build_matrix(word_index, embeddings_index):
    cols = EMBED_SIZE
    rows = len(word_index) + 1
    embedding_matrix = np.zeros(rows, cols)

    for word, i in word_index.items():
        try: embedding_matrix[i] = embeddings_index[word]
        except: embedding_matrix[i] = embeddings_index["unknown"]

    return embedding_matrix
