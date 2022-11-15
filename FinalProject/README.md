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

