# Based on this example: https://pythonnlp.quora.com/Generating-Random-Texts-with-NLTK

import nltk
from nltk.probability import ConditionalFreqDist
from nltk.corpus import genesis
import random

fname = "titles/illinois.txt"


with open(fname) as f:
    titles = f.read().splitlines()

tokens = nltk.word_tokenize(" ".join(titles))

text = nltk.Text(tokens)

def cfd(text, maxhistory):
    ngrams = []
    for i in range(2, maxhistory + 1):
        ngrams += nltk.ngrams(text, i)
    return ConditionalFreqDist([(tuple(a), b) for *a,b in ngrams])


def generate(seed, cfd, maxhistory, maxcount=10):
    for i in range(maxcount):
        for j in range(maxhistory-1, 0, -1):
            if tuple(seed[-j:]) in cfd:
                valuesum=sum(cfd[tuple(seed[-j:])].values())
                value=random.randint(0, valuesum)
                for key in cfd[tuple(seed[-j:])].keys():
                    value-=cfd[tuple(seed[-j:])][key]
                    if value <= 0:
                        seed.append(key)
                        break
                break
            else:
                continue
    return seed

maxhistory = 5
cfd = cfd(text, maxhistory)
generated = generate(['Hawk'], cfd, maxhistory)

print(" ".join(generated))
