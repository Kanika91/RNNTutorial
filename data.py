import torch
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import operator

#Read the training file for MovieLens Dataset

class SequenceGenerator(object):

    def __init__(self, dataset, rootFolder, trainFile, testFile):
        self.dataset = dataset
        self.rootFolder = rootFolder
        self.trainFile = trainFile
        self.testFile = testFile

    def getUserSequenes(self):
        #Read train and test file and create sequence
        train_sequences = []
        Users = []
        Items = set()
        history = dict()
        with open(self.rootFolder + "/" + self.trainFile, "r") as f:
            for line in f:
                user, item, rating, timestamp = line.strip().split("\t")
                user = int(user)
                if user not in Users and len(Users)!= 0:
                    sorted_history = sorted(history.items(), key=operator.itemgetter(1))
                    sorted_history = [i[0] for i in sorted_history]
                    train_sequences.append(list(sorted_history[:60]))
                    history = {}
                    Users.append(user)
                else:
                    if len(Users) == 0:
                        Users.append(int(user))
                    history[int(item)] = int(timestamp)
                    Items.add(int(item))

            if len(history) > 0:
                sorted_history = sorted(history.items(), key=operator.itemgetter(1))
                sorted_history = [i[0] for i in sorted_history]
                train_sequences.append(list(sorted_history[:60]))

        test_sequences = dict()
        with open(self.rootFolder + "/" + self.testFile, "r") as f:
            for line in f:
                user, item, rating, timestamp = line.strip().split("\t")
                user = int(user)
                item = int(item)
                if user not in test_sequences:
                    test_sequences[user] = set()
                test_sequences[user].add(item)

        return train_sequences, Users, Items, test_sequences
