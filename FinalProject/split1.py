import random

with open('cleaned_corpus.txt') as corpus:
    # read lines into list
    lines = corpus.readlines()
    corpus_size = len(lines)
    # shuffle list
    shuffled_corpus = random.sample(lines, corpus_size)
    # split shuffled list into two sublists
    first_half = shuffled_corpus[:int(corpus_size/2)]
    second_half = shuffled_corpus[int(corpus_size/2):]

with open('pos_paraphrase_examples.txt', 'w') as true_corpus:
    for line in first_half:
        true_corpus.write(line)

with open('neg_paraphrase_examples.txt', 'w') as false_corpus:
    left_sentences = []
    right_sentences = []
    for line in second_half:
       split_pair = line.split('\t')
       left_sentences.append(split_pair[0])
       right_sentences.append(split_pair[1].rstrip())
    right_sentences.append(right_sentences.pop(0)) 
    for i in range(len(left_sentences)):
       false_corpus.write(f'{left_sentences[i]}\t{right_sentences[i]}')
