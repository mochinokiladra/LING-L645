from math import log
import numpy as np
import json
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence

    max_T, max_A = data.shape

    # Loop over time
    for t in range(max_T):
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # Loop over possible alphabet outputs
            for c in range(max_A - 1):
                candidate = [seq + [c], score - log(data[t, c])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

f = open('output.json')
data = json.load(f)

beam_width = 3
logits_data = np.array(data['logits'])
alphabet_data = data['alphabet']

# decode sequence
result = beam_search_decoder(logits_data, beam_width)
# print result
'''
# prints all results without collapsing repeated characters:
for i, seq in enumerate(result):
    print(''.join([alphabet_data[s] for s in seq[0]]))
'''

# collapses repeated characters:
message = ' '
for i, seq in enumerate(result):
  for s in seq[0]:
    if message[-1] != alphabet_data[s]:
       message = message + alphabet_data[s]
  message = message + '\n'
message = message[1:]

print(message)

# produce heatmap
fig, ax = plt.subplots()
im = ax.imshow(logits_data)
fig.tight_layout()
plt.show()
f.close()
