import sys, re
import numpy as np
import math

from trigram_model import *

###############################################################################

training_samples = []
vocabulary = set(['<UNK>'])
line = sys.stdin.readline()
while line:
    tokens = preprocess(line)
    for i in tokens: vocabulary.add(i) 
    training_samples.append(tokens)
    line = sys.stdin.readline()

word2idx = {k: v for v, k in enumerate(vocabulary)} # word : count
idx2word = {v: k for k, v in word2idx.items()} # count : word

x_train = []
y_train = []
for tokens in training_samples:
    for i in range(len(tokens) - 2): #...# 
        x_train.append([word2idx[tokens[i] + ' ' + tokens[i+1]]]) #!!!#
        y_train.append([word2idx[tokens[i+2]]]) #!!!#

x_train = np.array(x_train)
y_train = np.array(y_train)

BATCH_SIZE = 1
NUM_EPOCHS = 10

train_set = np.concatenate((x_train, y_train), axis=1)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

loss_function = nn.NLLLoss()
model = TrigramNNmodel(len(vocabulary), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(NUM_EPOCHS):
    for i, data_tensor in enumerate(train_loader):
        context_tensor = data_tensor[:,0:1] #!!!#
        target_tensor = data_tensor[:,1] #!!!#

        model.zero_grad()

        log_probs = model(context_tensor)
        loss = loss_function(log_probs, target_tensor)

        loss.backward()
        optimiser.step()    

    print('Epoch:', epoch, 'loss:', float(loss))

torch.save({'model': model.state_dict(), 'vocab': idx2word}, 'trigram_model.lm')

print('Model saved.')
