from sys import argv

def get_pair_tuple(corpus, line):
   line = int(line)
#   f = open(file_name)
   split_pair = corpus[line].split('\t')
   para_pair = (split_pair[0],split_pair[1].rstrip())
#   f.close()
   return para_pair

