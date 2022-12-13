# splits the cleaned paraphrase corpus into three parts: positive, negative, and (for now) positive examples that will be turned into the difficult negative examples in a different script
import random

with open('cleaned_corpus.txt') as corpus:
    # read lines into list
    lines = corpus.readlines()
    corpus_size = len(lines)
    # shuffle list
    shuffled_corpus = random.sample(lines, corpus_size)
    # split shuffled list into three sublists
    third_of_corpus = int(corpus_size/3)
    print(third_of_corpus)
    first_third = shuffled_corpus[:third_of_corpus]
    second_third = shuffled_corpus[third_of_corpus:third_of_corpus*2]
    third_third = shuffled_corpus[third_of_corpus*2:]

# create a corpus of positive paraphrase examples (true paraphrases):
with open('pos_paraphrase_examples.txt', 'w') as true_corpus:
    for line in first_third:
        true_corpus.write(line)

# create a corpus of paraphrases to make "difficult" examples of with generate_neg_examples script:
with open('diff_paraphrase_examples.txt', 'w') as diff_corpus:
    for line in second_third:
        diff_corpus.write(line)

# create a corpus of negative paraphrase examples (random non-paraphrase sentence pairs):
with open('neg_paraphrase_examples.txt', 'w') as false_corpus:
    left_sentences = []
    right_sentences = []
    for line in third_third:
       split_pair = line.split('\t')
       left_sentences.append(split_pair[0])
       right_sentences.append(split_pair[1])
    right_sentences.append(right_sentences.pop(0)) 
    for i in range(len(second_third)):
       pair = left_sentences[i] + '\t' + right_sentences[i]
       false_corpus.write(pair)
