import sys, math, re, pickle
from collections import defaultdict, Counter

def tokenize(s):
    tokens = r'<BOS>|\w+|\.|!|\?|\,'
    return re.findall(tokens, s)

def create_model():
    model = defaultdict(lambda : defaultdict(float)) 
    unigrams = []

    line = sys.stdin.readline()
    while line: # Collect counts from standard input
        # !!! Collect unigram counts !!!
        text = ''
        line = '<BOS> ' + line
        text = text + line.rstrip()
        unigram = tokenize(text)
        [unigrams.append(u) for u in unigram]
        line = sys.stdin.readline()

    unigrams_count = Counter(unigrams)
    # account for unknown words:
#    unigrams_count['<UNK>'] = 1 

    # print(unigrams_count)

    # !!! Now calculate the probabilities !!!
    def unigram_prob(unigram):
        return unigrams_count[unigram] / len(unigrams_count)

    for k,v in unigrams_count.items():
        model[k] = unigram_prob(k)

    print('Saved %d unigrams.' % sum([len(i) for i in model.items()]))
    pickle.dump(dict(model), open('unigram_model.lm', 'wb'))

if sys.stdin:
    create_model()
