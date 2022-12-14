import sys, math, re, pickle
from collections import defaultdict, Counter

def tokenize(s):
    tokens = r'<BOS>|\w+|\.|!|\?|\,'
    return re.findall(tokens, s)

def create_model():
    model = defaultdict(lambda : defaultdict(float)) 
    unigrams = []
    bigrams = []

    line = sys.stdin.readline()
    while line: # Collect counts from standard input
        # !!! Collect bigram and unigram counts !!!
        text = ''
        line = '<BOS> ' + line
        text = text + line.rstrip()
        unigram = tokenize(text)
        [unigrams.append(u) for u in unigram]
        [bigrams.append((unigram[i], unigram[i+1])) for i in range(len(unigram[:-1]))]
        line = sys.stdin.readline()

    unigrams_count = Counter(unigrams)
    # account for unknown words:
#    unigrams_count['<UNK>'] = 1 
    bigrams_count = Counter(bigrams) 

    # smoothing: add 1 to the count of all bigrams in the corpus
    for k,v in unigrams_count.items():
        for l,w in unigrams_count.items():
            if (k,l) not in bigrams_count:
                bigrams_count[(k,l)] = 1
    for k,v in bigrams_count.items():
        bigrams_count[k] += 1
        
    # print(unigrams_count)
    # print(bigrams_count)

    # !!! Now calculate the probabilities !!!
    def bigram_prob(bigram):
         return bigrams_count[bigram] / (unigrams_count[bigram[0]] + len(unigrams_count))


    for k,v in bigrams_count.items():
        model[k[0]][k[1]] = bigram_prob((k[0],k[1]))
#        print(k, bigram_prob((k[0],k[1])))

    print('Saved %d bigrams.' % sum([len(i) for i in model.items()]))
    pickle.dump(dict(model), open('model.lm', 'wb'))

if sys.stdin:
    create_model()
