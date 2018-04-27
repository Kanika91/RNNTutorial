from data import SequenceGenerator
import time
import torch
import torch.nn as nn
from model import RNN
from torch.autograd import Variable
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hyper Parameters
num_layers = 1
hidden_size = 50
num_epochs = 3
learning_rate = 0.1
clip = 0.25
USE_CUDA = False
seed = 1234734614
torch.manual_seed(seed)
if USE_CUDA:
    torch.cuda.manual_seed(seed)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(num_items):
    correct = 0
    hidden = rnn.init_hidden()

    for user in test_sequences:
        sequence = train_sequences[user-1]
        input_variable = Variable(torch.LongTensor(sequence))
        output, hidden = rnn(input_variable, hidden)
        output = output[-1, :]
        hidden = repackage_hidden(hidden)
        target = test_sequences[user]

        topv, topi = output.data.topk(len(target))
        predicted = set(topi[:len(target)])
        if len(predicted.intersection(target)) > 0:
            correct+=1

    acc = correct/ float(len(test_sequences))
    print "Test set accuracy", acc
    return acc

#Load dataset
Folder = '/Users/kanika/Documents/ml-100k/'
SG = SequenceGenerator("MovieLens", Folder, trainFile='u1.base', testFile='u1.test')
train_sequences, Users, Items, test_sequences = SG.getUserSequenes()
print "#TrainSequence, #Users, #Items", len(train_sequences), len(Users), max(Items)

#Load the model
rnn = RNN(max(Items), hidden_size, num_layers)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad,rnn.parameters()), lr=learning_rate)

accuracy = 0
epoch_loss = []
epoch_accuracy = []
train_accuracy = []
epochs = []
try:
    #Call train on the model
    for epoch in range(1, num_epochs + 1):

        epochs.append(epoch)
        hidden = rnn.init_hidden()
        loss_total = 0
        acc = 0
        # Get training data for this cycle
        for i, sequence in enumerate(train_sequences):

            input_variable = Variable(torch.LongTensor(sequence[:-1]))
            targets = sequence[1:]
            target_variable = Variable(torch.LongTensor(targets))

            hidden = repackage_hidden(hidden)
            rnn.zero_grad()
            output, hidden = rnn(input_variable, hidden)
            loss = criterion(output, target_variable.contiguous().view(-1))

            val = (target_variable.data.view(-1).eq(torch.max(output, 1)[1].data).sum())
            acc+= (val/float(len(output.data)))
            loss.backward()

            torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)
            optimizer.step()

            # Keep track of loss
            #print("Loss for this step is", loss)
            loss_total += loss.data[0]

        if epoch > 0:
            acc = acc/float(i+1)
            print("Total loss for epoch", epoch, loss_total)
            print "Train accuracy ", acc
            epoch_loss.append(loss_total)
            train_accuracy.append(acc)

	    sys.stdout.flush()
        curr_accuracy = evaluate(max(Items)+1)
        epoch_accuracy.append(curr_accuracy)
        if curr_accuracy > accuracy:
           accuracy = curr_accuracy
           with open('model.pt', 'wb')   as f:
               torch.save(rnn, f)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

plt.close("all")
plt.plot(epoch_loss)
plt.ylabel("Total Loss")
plt.xlabel("Epochs")
plt.savefig('epoch_loss.png')
plt.clf()
#plt.show()

plt.plot(epochs, epoch_accuracy, 'r', label='Test')
plt.plot(epochs, train_accuracy, 'b', label='Train')
plt.ylabel("Train & Test accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.savefig('epoch_acc.png')
#plt.show()
plt.clf()

# Load the best saved model.
with open('model.pt', 'rb') as f:
    rnn = torch.load(f)

print("Best accuracy is ")
evaluate(max(Items)+ 1)
