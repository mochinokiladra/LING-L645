# Improving Paraphrase Detection Using Lexically Similar Non-Paraphrastic Sentence Pairs

The goal of this project was to investigate whether a paraphrase detection model might be improved by giving it more "challenging" non-paraphrase sentence pairs. By "challenging," I mean sentences that are similar on the surface (such as having many lexical items in common) such that they might confuse a computer, but that a human could easily tell are not semantically equivalent.

I used the cosine similarity of vectorized sentences (with vectors created from word counts) from the corpus to select the "challenging" sentence pairs. The idea was that these sentence pairs would have a lot of lexical overlap, potentially making it difficult for a system to recognize them as non-paraphrases.

My objective was to answer the following questions:
1. Are non-paraphrastic sentence pairs with a high cosine similarity more difficult for a machine learning model to classify than sentences that are randomly paired? How much more difficult?
2. Does training a model on these more difficult examples help it improve its ability to distinguish paraphrases from non-paraphrases?

Here is the short version of what I did:

* I downloaded a corpus of paraphrase pairs and divided it into three groups: 
  * one with sentence pairs that were actually paraphrases
  * one with randomly paired sentences that were assumed not to be paraphrases
  * one with sentences that were paired with other sentences in the corpus with which they had a high cosine similarity
* I created two training sets and two test sets from these three groups of sentence pairs
* I used a BERT-based model to run some experiments and answer the above two questions.

I used the `distilbert-base-uncased model` from [Huggingface](https://huggingface.co/models) to carry out my experiments.

### What is paraphrase detection?
First of all, what does paraphrase detection entail? Well, before we get to that, maybe we should have a working definition of "paraphrase." If Sentence B is a paraphrase of Sentence A, then the following ought to be true:
1. Sentence A and Sentence B are not the same sentence on the surface.
2. Sentence A and Sentence B have (approximately) the same underlying meaning.

Sentence A might differ syntactically from Sentence B, such as in the following example:

> 1a) The field of computational linguistics is fascinating, in my opinion.

> 1b) In my opinion, the computational linguistics field is fascinating.

Alternatively, the two sentences could have lexical differences:

> 2a) The field of computational linguistics is fascinating, in my opinion.

> 2b) The subject of computational linguistics is very interesting, in my view.

Often, paraphrases incorporate both lexical and syntactic changes:

> 3a) The field of computational linguistics is fascinating, in my opinion.

> 3b) I find the computational linguistics field to be very interesting.

A paraphrase detection system should be able to tell you that 3a and 3b are paraphrases, but that 4a and 4b are not (even though the latter two have more words in common):

> 4a) The field of computational linguistics is fascinating, in my opinion.

> 4b) The field of beautiful flowers is breathtaking, in my opinion.

### The Data Source
* The datasets used in this project came from the [ParaNMT-50M](https://aclanthology.org/P18-1042.pdf) corpus. 
* More specifically, I used the "Para-nmt-5m-processed dataset" from John Wieting’s [web page](https://www.cs.cmu.edu/~jwieting/). 
 * This is a subset of the corpus, containing 5,370,128 paraphrase pairs that have been pre-tokenized and fully lowercased. 
* The ParaNMT-50M corpus was created by taking several Czech-English parallel corpora consisting of human translations and translating the Czech side to English using a neural machine translation system. This is similar to the popular “pivoting” method, in which sentences from one language are translated into another language and then back-translated into the original language to create artificial paraphrases.

### Data Processing Steps
Here are all of the things I did with the data before using it for my experiments:
* When I started working with the data, I realized there were some lines in the corpus that were not actual sentences. For instance, one of the lines read:
> v5.9.1             &emsp; &emsp;            v5.9.1 

This wasn't ideal, because I wanted to use actual sentences for this project as much as possible. Furthermore, some of the functions in scikit-learn apparently won’t process inputs that don’t resemble actual words. So I removed all lines in the corpus that did not contain any consecutive alphabetic characters using the `clean_corpus.py` script. This brought the corpus size down to 4,346,278 pairs.
* I wrote a script to calculate the average cosine similarity between the sentence pairs in the corpus. This script is called `cossim.py`. It first creates a representation of each sentence using the `count_vectorizer` function from [scikit-learn](https://scikit-learn.org/stable/) that is based on word counts. It then makes use of the `cosine_similarity()` function from scikit-learn. To run this script from the command line, you can type `python3 cossim.py [filename]`. The lines of the input file need to be tab-separated sentence pairs. This is the format that the processed Para-NMT data comes in.
 * The average cosine similarity between pairs in the corpus was found to be ~0.5740.
* I then wrote another script, `split1.py`, to split the corpus into two parts: one with true paraphrase pairs and one with randomly paired sentences (and during the pairing process, pairing with a sentence’s original counterpart was blocked, so none of the resulting pairs should be paraphrases). This script just runs by typing `python3 split1.py`. The filenames I wanted to use are hardcoded in; it was meant to be single use.
* I then ran `cossim.py` on both of these subcorpora. I found that the average cosine similarity of the true paraphrase pairs was extremely close to that of the original corpus, ~0.5740. The average cosine similarity of the randomly paired sentences was ~0.0493. Paraphrases in this corpus, as expected, have a much higher average cosine similarity than randomly paired sentences. However, there were many "true" paraphrases that had a low cosine similarity, even as low as 0.0.
* The file containing true paraphrase pairs was further split into two datasets, one of which was used to generate a more “challenging” set of non-paraphrase sentence pairs. 
 * The script `generate_neg_examples.py` takes a file containing paraphrase pairs, and for each sentence on the left-hand side of the corpus, it searches for sentences with which this sentence has a high cosine similarity (set at > 0.56, based on the finding that true paraphrases had a cosine similarity of around 0.57 on average). When a sentence has a list of candidates, it chooses the best of those candidates that hasn’t yet been paired with another sentence. If there are no candidates that haven’t been paired yet, the sentence simply gets thrown out. I only wanted to have pairs that met the criterion of having a cosine similarity > 0.56. 
 * Running this script yielded 172,811 pairs. However, inspection of the pairs revealed that there were some duplicates among the right-hand matches due to an error in my code. Oops! Removing the duplicates resulted in 94,588 valid pairs. The average cosine similarity in this set ended up being ~0.6064. 
* I then created a subcorpus of 200,000 true paraphrase pairs and 100,000 randomly paired non-paraphrases. After that, I wrote another script, `generate_tsv_data.py`, to generate .tsv files containing positive and negative sentence pairs along with their labels (1 for paraphrase and 0 for non-paraphrase). I used this script to generate two training sets: `train_1.tsv` and `train_2.tsv`, and two test sets: `test_1.tsv` and `test_2.tsv`. The true paraphrase examples in each of the training sets are the same, but the non-paraphrase examples are different; `train_1` contains randomly paired sentences, and `train_2` contains the sentences that were paired based on high cosine similarity. The test sets are organized in the same way (obviously with different examples than the training sets). 
 * To run `generate_tsv_data.py`: type `python3 [filename with paraphrases], [filename with non-paraphrase pairs], [output filename (with .tsv file extension)]`
* Finally, I uploaded my datasets to Huggingface so that I could easily access them for my experiments.

### Experiments and Results
In a Google Colab notebook, I did two separate experiments using a distilBERT model, which are described below. You can view the code and output from these experiments in `experiment_01.ipynb` and `experiment_02.ipynb`. 

First, I trained distilbert-base-uncased on train_1 and then tested the model on each of the test sets. Remember, this is the training set that used randomly paired sentences as its non-paraphrase examples. I expected that this model would do quite well on the `test_1` set since it was also made up of true paraphrases and randomly paired sentences, and I expected it to struggle more with `test_2` (the one where the non-paraphrase pairs had a high cosine similarity). 

The F1 scores for `test_1` and `test_2` are reported below:


| Test Set 1 (random pairs) |  Test Set 2 (high cosine similarity pairs) |
|:-----------------------   |:----------------------                     |
| 0.9987                    | 0.8902                                     |

From these F1 scores, we can see that high cosine similarity pairs do make the task significantly more difficult for the model.

Experiment 2 was the same as Experiment 1, except that it used train_2 to train the model. I thought that training the model on high-cosine-similarity pairs might help it more accurately predict paraphrases.

Results from Experiment 2 (F1 score):
| Test Set 1 (random pairs) |  Test Set 2 (high cosine similarity pairs) |
|:-----------------------   |:----------------------                     |
| 0.9869                    | 0.9774                                     |

So, training on high cosine similarity pairs does lead to a significant improvement in its ability to distinguish paraphrases from similar-looking non-paraphrases. Interestingly, the model's performance on test set 1 decreases a bit. Test set 2 is still a bit more difficult than test set 1, even after training on the high-cosine-similarity pairs.

### What would have made this project better? / What could I try in the future?

#### Dataset Issues
* First of all, let’s talk about the dataset. In each sentence pair in ParaNMT-50M, there is one human-generated sentence and one machine-generated one that is assumed to be a paraphrase. There are obviously some problems with this. I think that an ideal paraphrase corpus would probably consist of pairs of high-quality human-created paraphrases with an abundance of lexical and syntactic variation. Unsurprisingly, that kind of thing isn't easy to find for cheap.

* I went with Para-NMT50 because even though it was created through backtranslation, it seemed to be of decent quality. It was easy to download and work with. It is also made up of sentences, and I was interested in doing something with sentential paraphrases. However, manual inspection of the corpus did reveal that quite a few sentences were not true paraphrases. There were apparent errors that probably came from the machine translations. There also appeared to be some sentences in the corpus that were entirely or mostly in Czech. I did not attempt to filter out any non-English sentences or erroneous translations.

* It would be interesting to try doing this with other corpora and see what the results are. It also might be worthwhile to adjust the Para-NMT dataset more: additional filtering out of problematic sentence pairs, for example.

* It would also be nice to try this on languages other than English. If it's difficult to find high-quality paraphrase corpora in English, it's even harder to find ones in other languages. However, there are some out there, like Opusparcus and the Tatoeba corpus, which contain paraphrase pairs in multiple languages.

#### Methodology
* In hindsight, I probably should have had another test set with a mix of both random sentence pairs and high cosine similarity pairs. Actually, I was definitely planning to do that at some point and then forgot about it as I was working on getting other parts of the project to work.
* It would have been nice to know the precision and recall for each experiment along with F1. If I were to redo the experiments, I would include them.
* It would be interesting to see how a higher cosine similarity threshold for the challenging sentence pairs would affect the results. Or to try using a different similarity metric.

#### Code Issues
* I need to go back and fix that bug in my code that paired the high-cosine-similarity sentences so that it will yield more sentence pairs. There's probably just a line in the wrong place. 




### References

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

John Wieting and Kevin Gimpel. 2018. ParaNMT-50M: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 451–462, Melbourne, Australia. Association for Computational Linguistics.
