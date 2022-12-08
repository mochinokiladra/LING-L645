TODO: Divide up dataset
-   Data is from ParaNMT-50 (or Para-paraNMT or Opusparcus?)
-   Take out a portion of the corpus data, split it into two parts (positive and negative examples)
-   For positive examples, just extract the paraphrase pairs.
-   Create 2-3 groups of negative examples:
    i.   one where the pairs are chosen at random (easy mode)
    ii.   one where the sentences are matched to sentences that have some surface similarity (such as having lexical items in common)
    iii.   one where sentences are matched to sentences that are similar by some metric of sentence similarity (but are not paraphrases)
           -   to do this one, first figure out what the avg. similarity between the true paraphrases is, and then try to find sentences that have something close to   that in the corpus

TODO: Train a model on detecting which pairs are paraphrases, using group i, ii, and iii of the negative examples, then run each model on test sets made up of group i, group ii, group iii, and mixed.

TODO: Evaluate model based on how many it got right.

Done so far:
- Cleaned some non-sentence data out of the corpus. For example, there were some pairs that were just something like "v5.9.1   v5.9.1". Did this using clean_corpus.py script. The result was a corpus of 4346278 sentence pairs.
- Ran script to calculate average cosine similarity between the pairs
        Result: 0.5739677709587692
- Partitioned corpus into two halves: pos_paraphrase_examples.txt and neg_paraphrase_examples.txt (using split1.py)
- Average cosine similarity in pos_paraphrase_examples: 0.5741301873537598
- Average cosine similarity in neg_paraphrase_examples: 0.04933681471483731
- Wrote a script to generate difficult negative examples (pairs have cosine similarity of > 0.56)


- Next step: fine-tune BERT to determine paraphrase pairs
- Maybe have the same number of pos, easy negative, and difficult negative examples?
