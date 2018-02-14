# Based on this example: https://pythonnlp.quora.com/Generating-Random-Texts-with-NLTK

import nltk
from nltk.probability import ConditionalFreqDist
from nltk.corpus import genesis

fname = "titles/illinois.txt"


with open(fname) as f:
    titles = f.read().splitlines()

tokens = nltk.word_tokenize(" ".join(titles))

text = nltk.Text(tokens)

ngrams = nltk.ngrams(text, 2)

def cfd(ngrams):
    return ConditionalFreqDist([(tuple(a), b) for *a,b in ngrams])

def generate(seed, cfd, maxcount=10):
    for i in range(maxcount):
        seed.append(cfd[tuple(seed[-1:])].max())
    return seed

print(generate(['Concerning'], cfd(ngrams)))
