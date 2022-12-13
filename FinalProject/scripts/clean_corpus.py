# remove lines that do not contain consecutive alphabetic characters
import re
f = 'para-nmt-5m-processed.txt'
corpus = open(f)
cleaned_corpus = open('cleaned_corpus.txt','w')
lines = corpus.readlines()
# clean_lines = []
for line in lines:
    if re.match('[A-Za-z][A-Za-z]+',line):
#        clean_lines.append(line.rstrip())
        cleaned_corpus.write(line)
cleaned_corpus.close()
corpus.close()
