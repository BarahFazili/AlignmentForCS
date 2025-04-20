import pickle
from collections import defaultdict
import re

def read_vocab(filename):
    vocab = defaultdict(lambda: 0)
    with open(filename, 'r') as f:
        for l in f.readlines():
            for word in (l.strip()).split():
                word = re.sub('[^A-Za-z0-9]+', '', word)
                vocab[word] += 1
    return vocab

vcb = read_vocab('test.txt')
# print(vcb)
with open('vcb_semeval.pkl', 'wb') as f:
    pickle.dump(dict(vcb), f)
