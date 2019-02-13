import gc
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from Attention import Attention
from Optim import Optim
from Beam import Beam


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        # h_0, c_0 = hidden
        h_0 = hidden
        c_0 = torch.zeros_like(hidden)
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        return input, (h_1, c_1)


class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        super().__init__()
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        self.input_size = opt.word_vec_size
        self.word_lut = nn.Embedding(dicts.size(), self.input_size, padding_idx=opt.PAD)
        self.rnn_1 = nn.GRU(self.input_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout,
                            bidirectional=opt.brnn)
        # R-NET
        self.attn = Attention(opt.rnn_size)
        self.rnn_2 = nn.GRU(opt.rnn_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout,
                            bidirectional=opt.brnn)
        self.linear_2 = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
        self.rnn_3 = nn.GRU(opt.rnn_size, self.hidden_size, num_layers=opt.layers, dropout=opt.dropout,
                            bidirectional=opt.brnn)
        self.linear_3 = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
        self.dropout = nn.Dropout(p=opt.dropout)

        # BIDAF
        self.BIDAF_similarity_linear = nn.Linear(opt.rnn_size * 3, 1, bias=False)
        self.BIDAF_c2q_sm = nn.Softmax(dim=1)
        self.BIDAF_q2c_sm = nn.Softmax(dim=0)
        self.BIDAF_rnn = nn.GRU(opt.rnn_size * 4, self.hidden_size, num_layers=opt.layers,
                                dropout=opt.dropout, bidirectional=opt.brnn)

    def load_pretrained_vectors(self, wv_tensor):
        # self.word_lut.weight.data.copy_(wv_tensor)
        self.word_lut.weight = nn.Parameter(wv_tensor, requires_grad=False)

    def BIDAF(self, p_inputs, q_inputs):
        # get similarity matrix
        Tp, B, D = p_inputs.size()
        Tq, _, _ = q_inputs.size()
        # Tp x Tq x B x D
        p_vec_1 = p_inputs.unsqueeze(1).expand(Tp, Tq, B, D)
        q_vec_1 = q_inputs.unsqueeze(0).expand(Tp, Tq, B, D)
        concat_vec = torch.cat((p_vec_1, q_vec_1, p_vec_1 * q_vec_1), -1)
        # Tp x Tq x B
        similarity_matrix = self.BIDAF_similarity_linear(concat_vec).squeeze(-1)

        # Context-to-query Attention
        c2q_similarity_matrix = self.BIDAF_c2q_sm(similarity_matrix)
        # Tp x Tq x B -> B x Tp x Tq, Tq x B x D -> B x Tq x D, return: B x Tp x D -> Tp x B x D
        c2q_vec = torch.bmm(c2q_similarity_matrix.transpose(1, 2).transpose(0, 1),
                            q_inputs.transpose(0, 1)).transpose(0, 1)

        # Query-to-context Attention
        q2c_similarity_matrix, _ = torch.max(similarity_matrix, dim=1)
        q2c_similarity_matrix = self.BIDAF_q2c_sm(q2c_similarity_matrix)
        # Tp x B -> B x Tp -> B x 1 x Tp, Tp x B x D -> B x Tp x D,
        # return: B x 1 x D -> B x Tp x D -> Tp x B x D
        q2c_vec = torch.bmm(q2c_similarity_matrix.transpose(0, 1).unsqueeze(1),
                            p_inputs.transpose(0, 1)).expand(B, Tp, D).transpose(0, 1)

        p_outputs = torch.cat((p_inputs, c2q_vec, p_inputs * c2q_vec, p_inputs * q2c_vec), -1)
        p_outputs, p_hn = self.BIDAF_rnn(p_outputs)
        return p_outputs, p_hn

    def forward(self, p_batch, q_batch, hidden=None):
        p_emb = self.word_lut(p_batch)
        p_outputs, p_hn = self.rnn_1(p_emb, hidden)
        q_emb = self.word_lut(q_batch)
        q_outputs, _ = self.rnn_1(q_emb, hidden)

        p_outputs, q_outputs = self.dropout(p_outputs), self.dropout(q_outputs)
        p_outputs, p_hn = self.BIDAF(p_outputs, q_outputs)

        # # R-NET
        # p_outputs, _ = self.attn(p_outputs, q_outputs)
        # p_outputs = torch.mul(p_outputs, torch.sigmoid(self.linear_2(p_outputs)))
        # p_outputs, p_hn = self.rnn_2(p_outputs)
        # p_outputs = self.dropout(p_outputs)
        # p_outputs, _ = self.attn(p_outputs, p_outputs)
        # p_outputs = torch.mul(p_outputs, torch.sigmoid(self.linear_3(p_outputs)))
        # p_outputs, p_hn = self.rnn_3(p_outputs)

        # p_outputs: (seq_len, batch, hidden_size*num_directions)
        return p_outputs, p_hn


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
        if decoder_mode == 0:
            return self.pointer_generator(p_batch, enc_hidden, context)
        elif decoder_mode == 1:
            return self.only_generator(p_batch, enc_hidden, context)
        elif decoder_mode == 2:
            return self.only_copy(p_batch, enc_hidden, context)
        else:
            print("Error")

    def pointer_generator(self, p_batch, enc_hidden, context):
        # p_batch: Tp x B, a_batch: Ta x B, enc_hidden: 2 x B x D, context: Tp x B x D
        # emb = self.word_lut(a_batch)  # Ta x B x D(300)
        Tp, B = p_batch.size()

        copy2total_map = torch.zeros(B, Tp, self.total_vocab_size)
        copy2total_map = self.toVariable(copy2total_map, requires_grad=False)  # B x Tp x dictSize
        copy2total_map = copy2total_map.scatter_(-1, p_batch.transpose(0, 1).contiguous().unsqueeze(-1), 1)

        zero_matrix = torch.zeros(B, self.total_vocab_size - self.gen_vocab_size)
        zero_matrix = self.toVariable(zero_matrix, requires_grad=False)  # B x (total-gen)dictSize

        cov_vec = torch.zeros(B, Tp)  # B x src_len
        cov_vec = self.toVariable(cov_vec, requires_grad=False)

        hn = enc_hidden  # h0: layer(2) x B x D(500)
        cn = torch.zeros(enc_hidden.size())  # c0: layer(2) x B x D(500)
        cn = self.toVariable(cn, requires_grad=False)

        context = context.repeat(1, self.opt.beam_size, 1) # Tp x beam_size*B x D
        copy2total_map = copy2total_map.repeat(self.opt.beam_size, 1, 1) # # beam_size*B x Tp x dictSize
        zero_matrix = zero_matrix.repeat(self.opt.beam_size, 1) # beam_size*B x (total-gen)dictSize
        cov_vec = cov_vec.repeat(self.opt.beam_size, 1) # beam_size*B x src_len
        hn = hn.repeat(1, self.opt.beam_size, 1) # layer(2) x beam_size*B x D(500)
        cn = cn.repeat(1, self.opt.beam_size, 1) # layer(2) x beam_size*B x D(500)

        beam = [Beam(self.opt, self.opt.beam_size) for k in range(B)]

        for it in range(self.opt.max_decoder_length):

            pre_word = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(1, -1) # 1 x beam_size*B
            emb_t = self.word_lut(pre_word)  # 1 x beam_size*B x D(300)

            output, (hn, cn) = self.rnn(emb_t, (hn, cn))  # output: 1 x beam_size*B x D(500)

            emb_t = emb_t.squeeze(0)  # beam_size*B x D(300)
            output = output.squeeze(0)  # beam_size*B x D(300)

            # context_vector: beam_size*B x D(500), attn: beam_size*B x src_len
            context_vector, attn, cov_vec = self.attn(output, context, cov_vec=cov_vec)

            # generator
            softmax_gen = self.gen_sm(self.gen_linear(context_vector))  # beam_size*B x dictSize
            if self.gen_vocab_size < self.total_vocab_size:
                softmax_gen = torch.cat((softmax_gen, zero_matrix), -1)

            # compute p_gen
            p_gen = torch.sigmoid(self.p_linear_1(context_vector) + self.p_linear_2(emb_t))  # beam_size*B x 1

            # pointer
            # attn = attn.transpose(0, 1) # beam_size*B x 1 x src_len
            attn = attn.unsqueeze(1)  # beam_size*B x 1 x src_len
            softmax_copy = torch.bmm(attn, copy2total_map)  # beam_size*B x 1 x dictSize
            softmax_copy = softmax_copy.squeeze(1)  # beam_size*B x dictSize

            # weighted sum
            output = torch.mul(p_gen, softmax_gen) + torch.mul(1 - p_gen, softmax_copy)  # beam_size*B x dictSize
            output = output.clamp(min=1e-12)  # 防止出现零
            output = torch.log(F.normalize(output, p=1, dim=-1))  # 标准化, beam_size*B x dictSize

            wordLk = output.view(self.opt.beam_size, B, -1).transpose(0, 1).contiguous() # B x beam_size x dictSize

            for i in range(B):
                done = beam[i].advance(wordLk.data[i])
                # layer(2) x beam_size*B x D(500)

                cov_vec_i = cov_vec.view(self.opt.beam_size, B, cov_vec.size(-1))[:, i]
                cov_vec_i.data.copy_(cov_vec_i.data.index_select(0, beam[i].getCurrentOrigin()))
                # cov_vec_new.data[:, i] = cov_vec_i.data.index_select(0, beam[i].getCurrentOrigin())

                hn_i = hn.view(-1, self.opt.beam_size, B, hn.size(-1))[:, :, i]
                hn_i.data.copy_(hn_i.data.index_select(1, beam[i].getCurrentOrigin()))
                # hn_new.data[:, :, i] = hn_i.data.index_select(1, beam[i].getCurrentOrigin())

                cn_i = cn.view(-1, self.opt.beam_size, B, cn.size(-1))[:, :, i]
                cn_i.data.copy_(cn_i.data.index_select(1, beam[i].getCurrentOrigin()))
                # cn_new.data[:, :, i] = cn_i.data.index_select(1, beam[i].getCurrentOrigin())

            if done:
                break

        for i in range(B):
            scores, ks = beam[i].sortBest() # beam_size
            scores = scores[:self.opt.n_best]  # n_best(1)
            ks = ks[:self.opt.n_best] # n_best(1)
            scores = scores.unsqueeze(0) # 1 x n_best(1)
            all_scores = scores if i == 0 else torch.cat((all_scores, scores), 0)  # B x n_best(1)
            for j, k in enumerate(ks):
                hyp_k = beam[i].getHyp(k).unsqueeze(0)  # 1 x max_decoder_length
                hyps = hyp_k if j == 0 else torch.cat((hyps, hyp_k), 0) # n_best(1) x max_decoder_length

            all_hyps = hyps if i == 0 else torch.cat((all_hyps, hyps), 0) # B x max_decoder_length

        # outputs: all_scores: n_best(1) x B, all_hyps: max_decoder_length x B
        return all_scores.t(), all_hyps.t()

    def only_generator(self, p_batch, enc_hidden, context):
        # p_batch: Tp x B, a_batch: Ta x B, enc_hidden: 2 x B x D, context: Tp x B x D
        # emb = self.word_lut(a_batch)  # Ta x B x D(300)
        Tp, B = p_batch.size()

        zero_matrix = torch.zeros(B, self.total_vocab_size - self.gen_vocab_size)
        zero_matrix = self.toVariable(zero_matrix, requires_grad=False)  # B x (total-gen)dictSize

        cov_vec = torch.zeros(B, Tp)  # B x src_len
        cov_vec = self.toVariable(cov_vec, requires_grad=False)

        hn = enc_hidden  # h0: layer(2) x B x D(500)
        cn = torch.zeros(enc_hidden.size())  # c0: layer(2) x B x D(500)
        cn = self.toVariable(cn, requires_grad=False)

        context = context.repeat(1, self.opt.beam_size, 1) # Tp x beam_size*B x D
        zero_matrix = zero_matrix.repeat(self.opt.beam_size, 1) # beam_size*B x (total-gen)dictSize
        cov_vec = cov_vec.repeat(self.opt.beam_size, 1) # beam_size*B x src_len
        hn = hn.repeat(1, self.opt.beam_size, 1) # layer(2) x beam_size*B x D(500)
        cn = cn.repeat(1, self.opt.beam_size, 1) # layer(2) x beam_size*B x D(500)

        beam = [Beam(self.opt, self.opt.beam_size) for k in range(B)]

        for it in range(self.opt.max_decoder_length):

            pre_word = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(1, -1) # 1 x beam_size*B
            emb_t = self.word_lut(pre_word)  # 1 x beam_size*B x D(300)

            output, (hn, cn) = self.rnn(emb_t, (hn, cn))  # output: 1 x beam_size*B x D(500)

            emb_t = emb_t.squeeze(0)  # beam_size*B x D(300)
            output = output.squeeze(0)  # beam_size*B x D(300)

            # context_vector: beam_size*B x D(500), attn: beam_size*B x src_len
            context_vector, attn, cov_vec = self.attn(output, context, cov_vec=cov_vec)

            # generator
            softmax_gen = self.gen_linear(context_vector)  # beam_size*B x dictSize
            if self.gen_vocab_size < self.total_vocab_size:
                softmax_gen = torch.cat((softmax_gen, zero_matrix), -1)
            output = softmax_gen
            output = self.LSm(output)

            wordLk = output.view(self.opt.beam_size, B, -1).transpose(0, 1).contiguous() # B x beam_size x dictSize

            for i in range(B):
                done = beam[i].advance(wordLk.data[i])
                # layer(2) x beam_size*B x D(500)

                cov_vec_i = cov_vec.view(self.opt.beam_size, B, cov_vec.size(-1))[:, i]
                cov_vec_i.data.copy_(cov_vec_i.data.index_select(0, beam[i].getCurrentOrigin()))
                # cov_vec_new.data[:, i] = cov_vec_i.data.index_select(0, beam[i].getCurrentOrigin())

                hn_i = hn.view(-1, self.opt.beam_size, B, hn.size(-1))[:, :, i]
                hn_i.data.copy_(hn_i.data.index_select(1, beam[i].getCurrentOrigin()))
                # hn_new.data[:, :, i] = hn_i.data.index_select(1, beam[i].getCurrentOrigin())

                cn_i = cn.view(-1, self.opt.beam_size, B, cn.size(-1))[:, :, i]
                cn_i.data.copy_(cn_i.data.index_select(1, beam[i].getCurrentOrigin()))
                # cn_new.data[:, :, i] = cn_i.data.index_select(1, beam[i].getCurrentOrigin())

            if done:
                break

        for i in range(B):
            scores, ks = beam[i].sortBest() # beam_size
            scores = scores[:self.opt.n_best]  # n_best(1)
            ks = ks[:self.opt.n_best] # n_best(1)
            scores = scores.unsqueeze(0) # 1 x n_best(1)
            all_scores = scores if i == 0 else torch.cat((all_scores, scores), 0)  # B x n_best(1)
            for j, k in enumerate(ks):
                hyp_k = beam[i].getHyp(k).unsqueeze(0)  # 1 x max_decoder_length
                hyps = hyp_k if j == 0 else torch.cat((hyps, hyp_k), 0) # n_best(1) x max_decoder_length

            all_hyps = hyps if i == 0 else torch.cat((all_hyps, hyps), 0) # B x max_decoder_length

        # outputs: all_scores: n_best(1) x B, all_hyps: max_decoder_length x B
        return all_scores.t(), all_hyps.t()

    def only_copy(self, p_batch, enc_hidden, context):
        # p_batch: Tp x B, a_batch: Ta x B, enc_hidden: 2 x B x D, context: Tp x B x D
        # emb = self.word_lut(a_batch)  # Ta x B x D(300)
        Tp, B = p_batch.size()

        copy2total_map = torch.zeros(B, Tp, self.total_vocab_size)
        copy2total_map = self.toVariable(copy2total_map, requires_grad=False)  # B x Tp x dictSize
        copy2total_map = copy2total_map.scatter_(-1, p_batch.transpose(0, 1).contiguous().unsqueeze(-1), 1)

        cov_vec = torch.zeros(B, Tp)  # B x src_len
        cov_vec = self.toVariable(cov_vec, requires_grad=False)

        hn = enc_hidden  # h0: layer(2) x B x D(500)
        cn = torch.zeros(enc_hidden.size())  # c0: layer(2) x B x D(500)
        cn = self.toVariable(cn, requires_grad=False)

        context = context.repeat(1, self.opt.beam_size, 1) # Tp x beam_size*B x D
        copy2total_map = copy2total_map.repeat(self.opt.beam_size, 1, 1) # # beam_size*B x Tp x dictSize
        cov_vec = cov_vec.repeat(self.opt.beam_size, 1) # beam_size*B x src_len
        hn = hn.repeat(1, self.opt.beam_size, 1) # layer(2) x beam_size*B x D(500)
        cn = cn.repeat(1, self.opt.beam_size, 1) # layer(2) x beam_size*B x D(500)

        beam = [Beam(self.opt, self.opt.beam_size) for k in range(B)]

        for it in range(self.opt.max_decoder_length):

            pre_word = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(1, -1) # 1 x beam_size*B
            emb_t = self.word_lut(pre_word)  # 1 x beam_size*B x D(300)

            output, (hn, cn) = self.rnn(emb_t, (hn, cn))  # output: 1 x beam_size*B x D(500)

            emb_t = emb_t.squeeze(0)  # beam_size*B x D(300)
            output = output.squeeze(0)  # beam_size*B x D(300)

            # context_vector: beam_size*B x D(500), attn: beam_size*B x src_len
            context_vector, attn, cov_vec = self.attn(output, context, cov_vec=cov_vec)

            # pointer
            # attn = attn.transpose(0, 1) # beam_size*B x 1 x src_len
            attn = attn.unsqueeze(1)  # beam_size*B x 1 x src_len
            softmax_copy = torch.bmm(attn, copy2total_map)  # beam_size*B x 1 x dictSize
            softmax_copy = softmax_copy.squeeze(1)  # beam_size*B x dictSize

            # weighted sum
            output = softmax_copy  # beam_size*B x dictSize
            output = output.clamp(min=1e-12)  # 防止出现零
            output = torch.log(F.normalize(output, p=1, dim=-1))  # 标准化, beam_size*B x dictSize

            wordLk = output.view(self.opt.beam_size, B, -1).transpose(0, 1).contiguous() # B x beam_size x dictSize

            for i in range(B):
                done = beam[i].advance(wordLk.data[i])
                # layer(2) x beam_size*B x D(500)

                cov_vec_i = cov_vec.view(self.opt.beam_size, B, cov_vec.size(-1))[:, i]
                cov_vec_i.data.copy_(cov_vec_i.data.index_select(0, beam[i].getCurrentOrigin()))
                # cov_vec_new.data[:, i] = cov_vec_i.data.index_select(0, beam[i].getCurrentOrigin())

                hn_i = hn.view(-1, self.opt.beam_size, B, hn.size(-1))[:, :, i]
                hn_i.data.copy_(hn_i.data.index_select(1, beam[i].getCurrentOrigin()))
                # hn_new.data[:, :, i] = hn_i.data.index_select(1, beam[i].getCurrentOrigin())

                cn_i = cn.view(-1, self.opt.beam_size, B, cn.size(-1))[:, :, i]
                cn_i.data.copy_(cn_i.data.index_select(1, beam[i].getCurrentOrigin()))
                # cn_new.data[:, :, i] = cn_i.data.index_select(1, beam[i].getCurrentOrigin())

            if done:
                break

        for i in range(B):
            scores, ks = beam[i].sortBest() # beam_size
            scores = scores[:self.opt.n_best]  # n_best(1)
            ks = ks[:self.opt.n_best] # n_best(1)
            scores = scores.unsqueeze(0) # 1 x n_best(1)
            all_scores = scores if i == 0 else torch.cat((all_scores, scores), 0)  # B x n_best(1)
            for j, k in enumerate(ks):
                hyp_k = beam[i].getHyp(k).unsqueeze(0)  # 1 x max_decoder_length
                hyps = hyp_k if j == 0 else torch.cat((hyps, hyp_k), 0) # n_best(1) x max_decoder_length

            all_hyps = hyps if i == 0 else torch.cat((all_hyps, hyps), 0) # B x max_decoder_length

        # outputs: all_scores: n_best(1) x B, all_hyps: max_decoder_length x B
        return all_scores.t(), all_hyps.t()


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
        all_scores, all_hyps = self.decoder(p_batch, a_batch, enc_hidden, context, decoder_mode=self.opt.decoder_mode)

        return all_scores, all_hyps