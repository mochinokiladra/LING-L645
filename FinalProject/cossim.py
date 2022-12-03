# takes a pair of sentences and outputs their cosine similarity

from sys import argv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import get_sentences

fname = 'cleaned_corpus.txt'

# takes as input a string that is a tab-separated sentence pair
# and returns their cosine similarity
def get_cosine_similarity(fname):
   f = open(fname)
   corpus = f.readlines()
   cossims = []
   for i in range(len(corpus)):
      sentences = get_sentences.get_pair_tuple(corpus, i)
      sent_1 = sentences[0]
      sent_2 = sentences[1]

      data = [sent_1, sent_2]

      count_vectorizer = CountVectorizer()

      vector_matrix = count_vectorizer.fit_transform(data)
      cossim = cosine_similarity(vector_matrix)[1][0]
      cossims.append(cossim)
      if i % 1000 == 0:
          print(i, 'lines processed.')
   f.close()
   return sum(cossims) / len(cossims)

print(get_cosine_similarity(fname))
