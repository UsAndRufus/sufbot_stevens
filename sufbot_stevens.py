import nltk

fname = "titles/illinois.txt"


with open(fname) as f:
    titles = f.read().splitlines()

tokens = nltk.word_tokenize(" ".join(titles))

text = nltk.Text(tokens)

model = nltk.ngrams(text, 3)

print(nltk.parse.generate(model, 5))
