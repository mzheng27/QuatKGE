import pickle
import bcolz
import numpy as np


glove_path = './QuatKGE/QuatE/SemanticTextClassifcation/CharE/data/'

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/840B.300.dat', mode='w')

voc = 2200000
counter = 0
with open(f'{glove_path}/glove.840B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        try:
            vect = np.array(line[-300:]).astype(np.float)
        except:
            print(np.array(line[1:]).shape)
            print(np.array(line[1:]))
            exit(1)
        vectors.append(vect)
        counter += 1
        #print(counter)
        if counter % (voc//100) == 0:
            print("{}% completed".format(counter*100//voc))

vectors = bcolz.carray(vectors[1:].reshape((-1, 300)), rootdir=f'{glove_path}/840B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/840B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/840B.300_idx.pkl', 'wb'))
