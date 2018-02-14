import nltk

fname = "titles/illinois.txt"


with open(fname) as f:
    titles = f.read().splitlines()

tokens = nltk.word_tokenize(titles[0])


print(titles[0])
print(tokens)
print(nltk.pos_tag(tokens))

text = nltk.Text(tokens)

print(text)
