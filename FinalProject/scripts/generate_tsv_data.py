# generates tsv file containing the data to be used in training and testing
# to run the script: first argument is the file containing the positive examples, second argument is file w/ negative examples, third argument is the output filename (with .tsv file extension)
import csv
from sys import argv

script,pos_file,neg_file,output_file = argv

data_file = open(output_file,'w')
data_writer = csv.writer(data_file, delimiter='\t')
data_writer.writerow(['Sentence1','Sentence2','Label', 'idx']) 

# bool_value is whether the sentence pairs in the input corpus are true paraphrases or not
def write_to_datafile(corpus_file, bool_value, i):
  corpus = open(corpus_file) 
  for line in corpus.readlines():
      spl = line.split('\t')
      left_line = spl[0]
      right_line = spl[1].rstrip()
      data_writer.writerow([left_line, right_line, bool_value, i]) 
      i += 1
  corpus.close()

# returns number of lines in a file:
def count_lines(inf):
    with open(inf) as f:
        c = len(f.readlines())
    return c

write_to_datafile(pos_file, 1, 0)
write_to_datafile(neg_file, 0, count_lines(pos_file)) 
