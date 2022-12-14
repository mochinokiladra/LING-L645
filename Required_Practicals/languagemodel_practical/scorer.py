import pickle
import sys
import math
import re

text = ''
test_list = []

def tokenize(s):
    tokens = r'<BOS>|\w+|\.|!|\?|\,'
    return re.findall(tokens, s)
tokens = 0
line = sys.stdin.readline()
while line:
    line = '<BOS> ' + line 
    test_list.append(line) 
    text = text + line
    line = sys.stdin.readline()
    tokens += len([tokenize(line)])

model = pickle.load(open('model.lm', 'rb'))

# returns the log score of a sentence
def log_scorer(sent):
    sent = tokenize(sent)
    sent_probs = []
    for w in range(len(sent)-1):
       bigram = (sent[w],sent[w+1])
       sent_probs.append(model[bigram[0]][bigram[1]])
    return sum([math.log(p) for p in sent_probs])

def sent_prob(sent):
    return math.exp(log_scorer(sent))

# input: list of sentences
# output: --
# prints each log probability, probability, and sentence
def display_results(sent_list):
    for sent in sent_list:
        print(f'{log_scorer(sent)}\t{sent_prob(sent)}\t{tokenize(sent)}')

display_results(test_list)

# input: a tokenized sentence
# output: perplexity
def bigram_perplexity():
    N = 0
    for line in test_list:
       N += len(line)
    sent_probs = sum([log_scorer(line) for line in test_list])
    perplexity = math.e ** ((-1/N)*sent_probs)
    return perplexity

unigram_model = pickle.load(open('unigram_model.lm', 'rb'))
def unigram_log_scorer(sent):
    sent = tokenize(sent)
    sent_probs = []
    for w in range(len(sent)):
       unigram = (sent[w])
       sent_probs.append(unigram_model[unigram])
    return sum([math.log(p) for p in sent_probs])

def unigram_sent_prob(sent):
    return math.exp(unigram_log_scorer(sent))

def unigram_perplexity():
    N = 0
    for line in test_list:
       N += len(line)
    sent_probs = sum([unigram_log_scorer(line) for line in test_list])
    perplexity = math.e ** ((-1/N)*sent_probs)
    return perplexity

print('bigram perplexity: ', bigram_perplexity())
print('unigram perplexity: ', unigram_perplexity())
