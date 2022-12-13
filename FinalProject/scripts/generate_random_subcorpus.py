# creates a random subcorpus
# example:
# python3 random_subcorpus.py file1 file2 500
# output would be a file2 with 500 random lines taken from file1
from sys import argv
import random
script,file1,file2,size = argv
with open(file1) as corpus:
    # read lines into list
    lines = corpus.readlines()
    # shuffle list
    subcorpus = random.sample(lines, int(size))

with open(file2, 'w') as new_corpus:
    for line in subcorpus:
        new_corpus.write(line)
