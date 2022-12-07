# generates "difficult" negative examples - i.e. negative paraphrase examples with a high cosine similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sys import argv

script,filename = argv

# read lines in file

# for each pair A,B:
# create new matrix of all the cosine similarities
# ignore the diagonals (those are true paraphrases)

# clear difficult_pairs.txt:
with open('difficult_pairs.txt','w') as of:
    of.write('')

with open(filename) as corpus:
   paraphrase_pairs = corpus.readlines()

vectorizer = CountVectorizer()

def cos_sim(sent1, sent2):
    X = vectorizer.fit_transform([sent1, sent2])
    return cosine_similarity(X.toarray())[0][1]

def generate_negative_examples(lines,output_file):
    length = len(lines) # the number of pairs in the corpus
    left_lines = []
    right_lines = []

    for line in lines:
        split_line = line.split('\t')
        left_lines.append(split_line[0])
        right_lines.append(split_line[1].rstrip())

    all_lines = left_lines + right_lines
    X = vectorizer.fit_transform(all_lines)
    left_vectors = X.toarray()[:length]
    right_vectors = X.toarray()[length:]

    candidate_pairs = {}

    for i in range(len(left_vectors)):
        for j in range(len(right_vectors)):
            if i != j:
               cossim = (cosine_similarity([left_vectors[i],right_vectors[j]])[0][1])
               if cossim > 0.56 and cossim < .9:
                   try:
                       candidates = candidate_pairs[left_lines[i]]
                       candidates.append(right_lines[j])
                   except:
                       candidates = [right_lines[j]]
                   candidate_pairs[left_lines[i]] = candidates

    # write out all of the candidates to a file to visualize them:
    def write_neg_candidates():
        with open('neg_example_candidates.txt','a') as neg_candidates:
            for k,v in candidate_pairs.items():
                neg_candidates.write(f'{k}:\n')
                for e in v:
                    neg_candidates.write(f'{v.index(e)+1}) {e}')
                    neg_candidates.write('\n')
                neg_candidates.write('\n')

    def pair_selection(of):
        matched_pairs = {}
        for k,v in candidate_pairs.items():
            remaining_candidates = []
            for c in v:
                if c not in matched_pairs.values():
                   remaining_candidates.append(c)
            try:
                if max([cos_sim(k,c) for c in remaining_candidates]):
                    matched_pairs[k] = c
            except:
                continue
        for k,v in matched_pairs.items():
            pair = k + '\t' + v + '\n'
            of.write(pair)

    pair_selection(output_file)


# split the corpus into batches so that it isn't too large for toarray:
num_pairs = len(paraphrase_pairs)
upper_limit = 1000
with open('difficult_pairs.txt','a') as of:
    i = 0
    while i+upper_limit < int(num_pairs):
        print(f"{i} out of {int(num_pairs)} lines processed...")
        generate_negative_examples(paraphrase_pairs[i:i+upper_limit],of)
        i += upper_limit 
    generate_negative_examples(paraphrase_pairs[i:],of)
