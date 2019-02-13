import torch
from tqdm import tqdm
import numpy


class Dict(object):
    def __init__(self, opt, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        self.opt = opt
        self.wv_list = None
        self.wv_tensor = None

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, totalVocabSize, genVocabSize):
        if totalVocabSize >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor([self.frequencies[i] for i in range(len(self.frequencies))])
        sorted_freq, idx = torch.sort(freq, 0, True)

        newDict = Dict(self.opt)
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:genVocabSize-len(self.special)]:
            tmp_idx = newDict.add(self.idxToLabel[i])
            newDict.frequencies[tmp_idx] = self.frequencies[i]
            # print(tmp_idx, self.idxToLabel[i], self.frequencies[i])

        for i in idx[genVocabSize-len(self.special):totalVocabSize-len(self.special)]:
            tmp_idx = newDict.add(self.idxToLabel[i])
            newDict.frequencies[tmp_idx] = self.frequencies[i]
            # print(tmp_idx, self.idxToLabel[i], self.frequencies[i])

        return newDict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels

    def load_word_embedding(self, path="./data/glove.840B.300d.txt", dim=300):
        # self.wv_list = [[0.0] * dim] * len(self.labelToIdx)
        self.wv_list = [[0.0 for _ in range(dim)] for _ in range(len(self.labelToIdx))]
        # self.wv_list = [[numpy.random.normal() for _ in range(dim)] for _ in tqdm(range(len(self.labelToIdx)))]
        n = 0
        with open(path, 'rb') as file:
            for line in tqdm(file):
                entries = line.strip().split(b' ')
                word, entries = entries[0], entries[1:]
                word = word.decode('utf-8')
                if word in self.labelToIdx:
                    n += 1
                    self.wv_list[self.labelToIdx[word]] = [float(x) for x in entries]
        print("number of words having embedding: ", n)
        # self.wv_tensor = torch.Tensor(self.wv_list)
        # if self.opt.usegpu:
        #     self.wv_tensor = self.wv_tensor.cuda()
        # return self.wv_tensor
        return self.wv_list