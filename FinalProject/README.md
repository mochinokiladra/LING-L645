The goal of this project was to test whether a paraphrase detection model might be improved by giving it more "challenging" non-paraphrase sentence pairs.

First of all, what does paraphrase detection entail? Well, before we get to that, maybe we should have a working definition of "paraphrase." If Sentence B is a paraphrase of Sentence A, then the following ought to be true:
1. Sentence A and Sentence B are not the same sentence (i.e., there should be some surface-level differences between them).
2. Sentence A and Sentence B should have (approximately) the same underlying meaning.

Sentence A might differ syntactically from Sentence B, such as in the following example:
1a) The field of computational linguistics is fascinating, in my opinion.
1b) In my opinion, the computational linguistics field is fascinating.
Alternatively, the two sentences could have lexical differences:
2a) The field of computational linguistics is fascinating, in my opinion.
2b) The subject of computational linguistics is very interesting, in my view.
Often, paraphrases incorporate both lexical and syntactic changes:
3a) The field of computational linguistics is fascinating, in my opinion.
3b) I find the computational linguistics field to be very interesting.

What "counts" as a pair of paraphrases might differ from person to person. According to some definitions, "I like dogs" and "I adore dogs" might be close enough in meaning to be considered paraphrases, but it can also be argued that "adore" deviates too much in meaning from "like" to accept this sentence pair as paraphrases. 

Data Source:
An ideal corpus for training a paraphrase detection system would consist of a very large number of high-quality human-generated paraphrases that have been annotated by additional humans. However, the construction of such a corpus would be extremely expensive. So, unsurprisingly, there isn't one like that. At least not one that is freely available. Various methods have been used to construct paraphrase corpora. A popular approach is a "pivoting" method in which sentences are translated using a machine translator and then translated back into the source language. By doing this, you end up with two sentences that are not identical but should, in theory, have the same meaning. Though there are obvious problems with this approach, it does generate a large number of paraphrase pairs that are mostly decent. And it isn't exorbitantly expensive. 

- Cleaned some non-sentence data out of the corpus. For example, there were some pairs that were just something like "v5.9.1   v5.9.1". Did this using clean_corpus.py script. The result was a corpus of 4346278 sentence pairs.
- Ran script to calculate average cosine similarity between the pairs
        Result: 0.5739677709587692
- Partitioned corpus into two halves: pos_paraphrase_examples.txt and neg_paraphrase_examples.txt (using split1.py)
- Average cosine similarity in pos_paraphrase_examples: 0.5741301873537598
- Average cosine similarity in neg_paraphrase_examples: 0.04933681471483731
- Wrote a script to generate difficult negative examples (pairs have cosine similarity of > 0.56)
- This script yielded 172811 pairs. However, inspection of the pairs revealed that there were some duplicates among the right-hand matches due to an error in my code. Removing the duplicates resulted in 94588 valid pairs with an average cosine similarity of 0.6063767811902363.
- Created subcorpora of 200k positive examples and 100k random pairs
- Wrote a script to generate .csv file (tab-separated) with pos and neg sentence pairs and expected value (1/0)
- Did two experiments using a distilBERT model:
-- First, trained distilBERT on 210000 sentence pairs labeled as paraphrase or non-paraphrase. In experiment 1, this used the randomly paired sentences as non-paraphrases. In experiment 2, this used the sentence pairs that were selected to have a high cosine similarity.
-- Tested the trained model on two datasets: one with randomly paired negative examples, one with only cosine-paired negative examples. In hindsight, it would have been a good idea to make another test set with a mix of random and cosine pairs. Well.
Results:
Experiment 1: {'f1': 0.9987428923466287, 'accuracy': 0.9983222222222222} {'f1': 0.8902130438331564, 'accuracy': 0.8325921902757505}
