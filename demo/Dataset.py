import math
import random
import torch
from torch.autograd import Variable
from Opt import Opt


class Dataset(object):
    def __init__(self, opt, dataset, batchSize, volatile=False):
        self.passage_idx = dataset['passage_idx']
        self.query_idx = dataset['query_idx']
        self.answer_idx = dataset['answer_idx']
        self.rouge_l = dataset['rouge_l']
        self.origin_answer = dataset['origin_answer']
        self.origin_passage = dataset['origin_passage']; self.origin_query = dataset['origin_query']
        self.query_id = dataset['query_id']; self.query_type = dataset['query_type']
        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.query_idx) / batchSize)
        self.volatile = volatile
        self.opt = opt

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(self.opt.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        pBatch = self._batchify(self.passage_idx[index * self.batchSize:(index + 1) * self.batchSize])
        qBatch = self._batchify(self.query_idx[index * self.batchSize:(index + 1) * self.batchSize])
        aBatch = self._batchify(self.answer_idx[index * self.batchSize:(index + 1) * self.batchSize])
        rBatch = self.rouge_l[index * self.batchSize:(index + 1) * self.batchSize]
        oaBatch = self.origin_answer[index * self.batchSize:(index + 1) * self.batchSize]
        opBatch = self.origin_passage[index * self.batchSize:(index + 1) * self.batchSize]
        oqBatch = self.origin_query[index * self.batchSize:(index + 1) * self.batchSize]
        qiBatch = self.query_id[index * self.batchSize:(index + 1) * self.batchSize]
        qtBatch = self.query_type[index * self.batchSize:(index + 1) * self.batchSize]



        # within batch sorting by decreasing length for variable length rnns
        # indices = range(len(pBatch))
        # batch = zip(indices, pBatch, qBatch, aBatch)
        # batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        # indices, pBatch, qBatch, aBatch = zip(*batch)
        def wrap(b):
            if b is None: return b
            # b = torch.stack(b,0).t().contiguous()
            b = b.t().contiguous()
            if self.opt.usegpu:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return wrap(pBatch), wrap(qBatch), wrap(aBatch), rBatch, \
               oaBatch, opBatch, oqBatch, qiBatch, qtBatch

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.passage_idx, self.query_idx, self.answer_idx, self.rouge_l,
                        self.origin_answer, self.origin_passage, self.origin_query,
                        self.query_id, self.query_type))
        self.passage_idx, self.query_idx, self.answer_idx, self.rouge_l, self.origin_answer, \
        self.origin_passage, self.origin_query, self.query_id, self.query_type = \
            zip(*[data[i] for i in torch.randperm(len(data))])