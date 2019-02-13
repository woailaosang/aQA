import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from Attention import Attention

class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers
        self.hidden_size = opt.rnn_size
        self.total_vocab_size = dicts.size()
        self.gen_vocab_size = min(opt.genVocabSize, self.total_vocab_size)

        self.word_lut = nn.Embedding(self.total_vocab_size, opt.word_vec_size, padding_idx=opt.PAD)
        # self.rnn = StackedLSTM(opt.layers, opt.word_vec_size, opt.rnn_size, opt.dropout)
        self.rnn = nn.LSTM(opt.word_vec_size, opt.rnn_size, opt.layers, dropout=opt.dropout)
        self.attn = Attention(opt.rnn_size, isCoverage=True)

        self.gen_linear = nn.Linear(opt.rnn_size, self.gen_vocab_size)
        self.gen_sm = nn.Softmax(dim=-1)

        self.p_linear_1 = nn.Linear(opt.rnn_size, 1, bias=False)
        self.p_linear_2 = nn.Linear(opt.word_vec_size, 1, bias=False)

        self.LSm = nn.LogSoftmax(dim=-1)
        # self.generator = nn.Sequential(nn.Linear(opt.rnn_size, dicts.size()), nn.LogSoftmax(dim=-1))

    def load_pretrained_vectors(self, wv_tensor):
        # self.word_lut.weight.data.copy_(wv_tensor)
        self.word_lut.weight = nn.Parameter(wv_tensor, requires_grad=False)

    def toVariable(self, vec, requires_grad=False):
        if self.opt.usegpu:
            vec = vec.cuda()
        return Variable(vec, requires_grad=requires_grad)

    def forward(self, p_batch, a_batch, enc_hidden, context, decoder_mode=0):
        # p_batch: Tp x B, a_batch: Ta x B, enc_hidden: 2 x B x D, context: Tp x B x D
        return self.pointer_generator(p_batch, a_batch, enc_hidden, context)


    def pointer_generator(self, p_batch, a_batch, enc_hidden, context):
        # p_batch: Tp x B, a_batch: Ta x B, enc_hidden: 2 x B x D, context: Tp x B x D
        emb = self.word_lut(a_batch) # Ta x B x D(300)
        Tp, B = p_batch.size()

        copy2total_map = torch.zeros(B, Tp, self.total_vocab_size)
        copy2total_map = self.toVariable(copy2total_map, requires_grad=False) # B x Tp x dictSize
        copy2total_map = copy2total_map.scatter_(-1, p_batch.transpose(0, 1).contiguous().unsqueeze(-1), 1)

        # zero_matrix = torch.zeros(1, B, self.total_vocab_size - self.gen_vocab_size)
        zero_matrix = torch.zeros(B, self.total_vocab_size - self.gen_vocab_size)
        zero_matrix = self.toVariable(zero_matrix, requires_grad=False) # B x (total-gen)dictSize

        cov_vec = torch.zeros(B, Tp) # B x src_len
        cov_vec = self.toVariable(cov_vec, requires_grad=False)

        hn = enc_hidden # h0: layer(2) x B x D(500)
        cn = torch.zeros(enc_hidden.size()) # c0: layer(2) x B x D(500)
        cn = self.toVariable(cn, requires_grad=False)

        # # emb:  Ta x B x D(300)
        # for it, emb_t in enumerate(emb.split(1, dim=0)):
        #     # emb_t: 1 x B x D(300)

        pre_word = a_batch[0] # B
        emb_t = self.word_lut(pre_word).unsqueeze(0) # 1 x B x D(300)
        p_gen_list = []
        for it in range(self.opt.max_decoder_length):
            # teacher forcing
            output, (hn, cn) = self.rnn(emb_t, (hn, cn)) # output: 1 x B x D(500)

            emb_t = emb_t.squeeze(0) # B x D(300)
            output = output.squeeze(0) # B x D(300)

            # context_vector: B x D(500), attn: B x src_len
            context_vector, attn, cov_vec = self.attn(output, context, cov_vec=cov_vec)
            # context_vector, attn = self.attn(output, context)

            # generator
            softmax_gen = self.gen_sm(self.gen_linear(context_vector)) # B x dictSize
            if self.gen_vocab_size < self.total_vocab_size:
                softmax_gen = torch.cat((softmax_gen, zero_matrix), -1)

            # compute p_gen
            p_gen = torch.sigmoid(self.p_linear_1(context_vector) + self.p_linear_2(emb_t)) # B x 1
            p_gen_list.append(p_gen.data[0][0])

            # pointer
            # attn = attn.transpose(0, 1) # B x 1 x src_len
            attn = attn.unsqueeze(1) # B x 1 x src_len
            softmax_copy = torch.bmm(attn, copy2total_map) # B x 1 x dictSize
            # softmax_copy = softmax_copy.transpose(0,1) # 1 x B x dictSize
            softmax_copy = softmax_copy.squeeze(1) # B x dictSize

            # weighted sum
            output = torch.mul(p_gen, softmax_gen) + torch.mul(1-p_gen, softmax_copy) # B x dictSize

            output = output.clamp(min=1e-12) # 防止出现零
            output = torch.log(F.normalize(output, p=1, dim=-1)) # 标准化, B x dictSize
            output = output.unsqueeze(0) # 1 x B x dictSize

            outputs = output if it==0 else torch.cat((outputs, output),0)


            pre_word = output.max(-1)[1] # 1 x B
            emb_t = self.word_lut(pre_word) # 1 x B x D(300)

        # outputs: Ta x B x dictSize
        return outputs, p_gen_list



class Model(nn.Module):
    def __init__(self, opt, encoder, decoder):
        super().__init__()
        self.opt = opt
        self.encoder = encoder
        self.decoder = decoder
        # self.generator = decoder.generator

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        result = context.data.new(*h_size).zero_()
        if self.opt.usegpu:
            result = result.cuda()
        return Variable(result, requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        # input: wrap(pBatch), wrap(qBatch), wrap(aBatch), rBatch, oaBatch
        p_batch = input[0]
        q_batch = input[1]
        a_batch = input[2][:-1]  # exclude last target from inputs
        context, enc_hidden = self.encoder(p_batch, q_batch)
        # init_output = self.make_init_decoder_output(context)
        enc_hidden = (self._fix_enc_hidden(enc_hidden))
        # out, dec_hidden, _attn = self.decoder(p_batch, a_batch, enc_hidden, context, init_output)
        out, p_gen_list = self.decoder(p_batch, a_batch, enc_hidden, context, decoder_mode=self.opt.decoder_mode)

        return out, p_gen_list

def toLabel(li, add_stop_token=None):
    stop_token = ['<pad>', '</s>']
    if add_stop_token != None: stop_token = stop_token + add_stop_token
    stop_idx = [dicts.lookup(token) for token in stop_token]
    result = []
    for idx in li:
        if idx in stop_idx:
            break
        else:
            result.append(dicts.idxToLabel[idx])
    return result

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import nltk
from Model_test import Encoder
from tqdm import tqdm
from Dataset import Dataset
from Opt import Opt
from myRouge import Rouge
from torch import cuda

opt = Opt()
opt.gpuid = 0
opt.batch_size = 1
opt.beam_size = 1
opt.decoder_mode = 0

dicts = torch.load("./data/preprocess_vocab.pt")
valid_dataset = torch.load("./data/preprocess_valid.pt")
validData = Dataset(opt, valid_dataset, opt.batch_size, volatile=True)

dicts.wv_tensor = torch.Tensor(dicts.wv_list)
if opt.usegpu:
    dicts.wv_tensor = dicts.wv_tensor.cuda()

m_pg = torch.load("./best_models/pg_e_12_ppl_2.75_BIDAF+PG+GPU1.mdl", map_location={'cuda:1':'cuda:0'})

encoder = Encoder(opt, dicts)
decoder = Decoder(opt, dicts)
model = Model(opt, encoder, decoder)
if opt.usegpu:
    cuda.set_device(opt.gpuid)
    model.cuda()
encoder.load_pretrained_vectors(dicts.wv_tensor)
decoder.load_pretrained_vectors(dicts.wv_tensor)
model.load_state_dict(m_pg)
model.eval()

def mrc(one_batch):
    passage_idx, query_idx, ref_answer_idx, ref_rouge_l, origin_answer, \
        origin_passage, origin_query, query_id, query_type = one_batch

    outputs, p_gen_list = model(one_batch)
    pre_answer = outputs.max(-1)[1].t().data.tolist()[0]
    pre_answer = toLabel(pre_answer)
    p_gen_list = p_gen_list[:len(pre_answer)]

    query_text = " ".join(toLabel(query_idx.t().data.tolist()[0]))

    p_copy_list = [1-item for item in p_gen_list]

    passage_word_list = toLabel(passage_idx.t().data.tolist()[0])
    passage_scores = [0.0 for _ in range(len(passage_word_list))]

    for i in range(len(passage_scores)):
        if passage_word_list[i] in pre_answer:
            passage_scores[i] = p_copy_list[pre_answer.index(passage_word_list[i])]


    highlight_passage = ""
    for word, value in zip(passage_word_list, passage_scores):
        word = 'UNK' if word == '<unk>' else word
        highlight_passage += ('<font style="background: rgba(255, 0, 255, %f)">%s</font>\n' % (value, word))
    highlight_answer = ""
    for word, value in zip(pre_answer, p_copy_list):
        word = 'UNK' if word == '<unk>' else word
        highlight_answer += ('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (value, word))

    return origin_query[0], origin_passage[0], highlight_passage, highlight_answer

def get_an_example():
    idx = random.randint(0, len(validData) - 1)
    one_batch = validData[idx]
    return mrc(one_batch)

def any_input(query, passage):
    one_data = {
        'passage_idx': [dicts.convertToIdx(nltk.word_tokenize(passage), opt.UNK_WORD, eosWord=opt.EOS_WORD)],
        'query_idx': [dicts.convertToIdx(nltk.word_tokenize(query), opt.UNK_WORD)],
        'answer_idx': [dicts.convertToIdx(nltk.word_tokenize('No Answer.'), opt.UNK_WORD, opt.BOS_WORD, opt.EOS_WORD)],
        'rouge_l': [0],
        'origin_answer': ['No Answer.'],
        'origin_passage': [passage],
        'origin_query': [query],
        'query_id': [0],
        'query_type': ['None']
    }
    one_batch = Dataset(opt, one_data, opt.batch_size, volatile=True)[0]
    return mrc(one_batch)
