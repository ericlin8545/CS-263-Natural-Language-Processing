import numpy as np

dim = 50
embeddings_index = {}
f = open("glove.twitter.27B.50d.txt", encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()