# Making a Better Paraphrase Detection by Using Similar Non-Paraphrases

The goal of this project was to test whether a paraphrase detection model might be improved by giving it more "challenging" non-paraphrase sentence pairs. I downloaded a corpus of paraphrase pairs and divided it into three groups: one with sentence pairs that were actually paraphrases, one with randomly paired sentences that were assumed not to be paraphrases, and one with sentences that were paired with other sentences in the corpus with which they had a high cosine similarity. The idea was that sentence pairs with a high cosine similarity would be harder for a system to distinguish between paraphrase and non-paraphrase because these sentences would appear to be similar on the surface even if they are not equivalent in meaning. 

### What is paraphrase detection?
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

### The Data Source
The datasets used in this project came from the ParaNMT-50M corpus (link). More specifically, I used the Para-nmt-5m-processed dataset from John Wieting’s web page. This is a subset of the corpus, containing 5,370,128 paraphrase pairs that have been pre-tokenized and fully lowercased. The ParaNMT-50M corpus was created by taking several Czech-English parallel corpora consisting of human translations and translating the Czech side to English using a neural machine translation system. This is similar to the popular “pivoting” method, in which sentences from one language are translated into another language and then back-translated into the original language to create artificial paraphrases.
When I started working with the data, I realized there were some lines in the corpus that were not actual sentences. For instance, one of the lines read:
v5.9.1   v5.9.1

### Data Processing Steps
Some of the functions in scikit-learn won’t process inputs that don’t resemble actual words. So I removed all lines in the corpus that did not contain any consecutive alphabetic characters using a clean_corpus.py script that I wrote. This brought the corpus size down to 4,346,278 pairs.
I wrote a script to calculate the average cosine similarity between the sentence pairs in the corpus. This script is called cossim.py. It makes use of the cosine_similarity() function from scikit-learn. To run this script from the command line, you can type python3 cossim.py [filename]. The lines of the input file need to be tab-separated sentence pairs. This is the format that the processed Para-NMT data comes in.
The average cosine similarity between pairs in the corpus was found to be ~0.5740.
I then wrote another script, split1.py, to split the corpus into two parts: one with true paraphrase pairs and one with randomly paired sentences (and during the pairing process, pairing with a sentence’s original counterpart was blocked, so none of the resulting pairs should be paraphrases). 
I then ran cossim.py on both of these subcorpora. I found that the average cosine similarity of the true paraphrase pairs was extremely close to that of the original corpus, ~0.5740. The average cosine similarity of the randomly paired sentences was ~0.0493. Paraphrases in this corpus, as expected, have a much higher average cosine similarity than randomly paired sentences. However, there were many paraphrases that had a low cosine similarity, even as low as 0.0.
The file containing true paraphrase pairs was further split into two datasets, one of which was used to generate a more “challenging” set of non-paraphrase sentence pairs. The script generage_neg_examples.py takes a file containing paraphrase pairs, and for each sentence on the left side of the corpus, it searches for sentences with which it has a high cosine similarity (a cosine similarity of > 0.56, based on the finding that true paraphrases had a cosine similarity of around 0.57 on average). When a sentence has a list of candidates, it chooses the best of those candidates that hasn’t yet been paired with another sentence. If there are no candidates that haven’t been paired yet, the sentence simply gets thrown out. I only wanted to have pairs that met the criterion of having a cosine similarity > 0.56. 
Running this script yielded 172,811 pairs. However, inspection of the pairs revealed that there were some duplicates among the right-hand matches due to an error in my code. Oops! Removing the duplicates resulted in 94,588 valid pairs. The average cosine similarity in this set ended up being ~0.6064. 
I then created a subcorpus of 200,000 true paraphrase pairs and 100,000 randomly paired non-paraphrases. After that, I wrote another script, generate_tsv_data.py, to generate .tsv files containing positive and negative sentence pairs along with their labels (1 for paraphrase and 0 for non-paraphrase). I used this script to generate two training sets: train_1.tsv and train_2.tsv, and two test sets: test_1.tsv and test_2.tsv. The true paraphrase examples in each of the training sets are the same, but the non-paraphrase examples are different; train_1 contains randomly paired sentences, and train_2 contains the sentences that were paired based on high cosine similarity. The test sets are organized in the same way (obviously with different examples than the training sets). I uploaded my datasets to Huggingface so that I could make use of their API for my experiments.

### Experiments and Results
I did two separate experiments using a distilBERT model, which are described below. You can view the code and output from these experiments in experiment_01.ipynb and experiment_02.ipynb. 

First, I trained distilbert-base-uncased on train_1 and then tested the model on each of the test sets. Remember, this is the training set that used randomly paired sentences as its non-paraphrase examples. I expected that this model would do quite well on the test_1 set since it was also made up of true paraphrases and randomly paired sentences, and I expected it to struggle more with test_2 (the one where the non-paraphrase pairs had a high cosine similarity). 
The f1 and accuracy scores from Experiment 1 are as follows:
Test set 1 (random pairs): 
F1: 0.9987	Accuracy: 0.9983
Test set 2 (high cosine similarity pairs):
F1: 0.8902	Accuracy: 0.8330
Experiment 2 was the same as Experiment 1, except that it used train_2 to train the model. I expected that training the model on high-cosine-similarity pairs would help it more accurately predict paraphrases.
Results from Experiment 2:
Test set 1 (random pairs):
F1: 0.9869	Accuracy: 0.9827
Test set 2 (high cosine similarity pairs):
F1: 0.9774	Accuracy: 0.9692

### What would have made this project better?
First of all, let’s talk about the dataset. In an ideal world, we would have perhaps a very large number of high-quality human-generated paraphrases that have been annotated for quality, meaning preservation, etc. But that’s expensive, hence the popularity of artificial methods of paraphrase generation such as backtranslation. There are obviously shortcomings to such approaches, and we can’t trust that all of the generated sentence pairs will be true paraphrases.
I went with Para-NMT50 because even though it was created through backtranslation, it seemed to be of decent quality. It is also made up of sentences, and I was interested in doing something with sentential paraphrases. However, manual inspection of the corpus did reveal that quite a few sentences were not true paraphrases. There were mismatches that probably came about from errors in the machine translations. There also appeared to be some sentences in the corpus that were entirely or mostly in Czech. I did not attempt to filter these out. So yeah, the model is getting some incorrect or unhelpful examples in training.
It would be interesting to try doing this with other corpora and see what the results are. It also might be worthwhile to adjust the Para-NMT dataset more—additional filtering out of problematic sentence pairs, for example, or a higher cosine similarity threshold for the “difficult” non-paraphrase examples that I used in train_2 and test_2. 
Code Issues
If I fix the bug in my code to match up the high-cosine-similarity sentence pairs, there will be more examples to work with. 
Other
In hindsight, I probably should have had another test set with a mix of both random sentence pairs and high cosine similarity pairs. Next time.
