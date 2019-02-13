import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, isCoverage=True):
        super().__init__()
        self.dim = dim
        self.isCoverage = isCoverage
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=False)
        self.linear_cov = nn.Linear(1, dim)
        self.v = nn.Linear(dim, 1, bias=False)

        self.linear_out = nn.Linear(dim*2, dim)
        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s, memory_lengths=None, cov_vec=None):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index `[batch x tgt_len x src_len]`
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        dim = self.dim
        wq = self.linear_query(h_t.contiguous().view(-1, dim))
        wq = wq.view(tgt_batch, tgt_len, 1, dim)
        wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

        uh = self.linear_context(h_s.contiguous().view(-1, dim))
        uh = uh.view(src_batch, 1, src_len, dim)
        uh = uh.expand(src_batch, tgt_len, src_len, dim)

        if self.isCoverage:
            # cov_vec: batch x src_len
            cov_vec = cov_vec.unsqueeze(-1) # batch x src_len x 1
            cv = self.linear_cov(cov_vec) # batch x src_len x dim
            cv = cv.view(src_batch, 1, src_len, dim)
            cv = cv.expand(src_batch, tgt_len, src_len, dim)
            wquh = self.tanh(wq + uh + cv) # batch x tgt_len x src_len x dim
        else:
            wquh = self.tanh(wq + uh) # batch x tgt_len x src_len x dim

        return self.v(wquh).squeeze(-1) # batch x tgt_len x src_len

    def forward(self, input, memory_bank, memory_lengths=None, cov_vec=None):
        """
        Args:
          input (`FloatTensor`): query vectors `[tgt_len x batch x dim]`
          memory_bank (`FloatTensor`): source vectors `[src_len x batch x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query `[tgt_len x batch x src_len]`
        """
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1) # batch x 1 x dim
        else:
            one_step = False
            input = input.transpose(0, 1)  # batch x tgt_len x dim

        memory_bank = memory_bank.transpose(0, 1) # batch x src_len x dim
        if memory_lengths is not None:  print("Error") # 暂时省略

        batch, sourceL, dim = memory_bank.size()
        _, targetL, _ = input.size()

        align = self.score(input, memory_bank, memory_lengths, cov_vec)
        # Softmax to normalize attention weights
        align_vectors = self.sm(align) # batch x tgt_len x src_len

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank) # batch x tgt_len x dim

        # concatenate
        concat_c = torch.cat([c, input], -1) # batch x tgt_len x (2*dim)
        attn_h = self.linear_out(concat_c) # batch x tgt_len x dim

        if one_step:
            attn_h = attn_h.squeeze(1) # batch x dim
            align_vectors = align_vectors.squeeze(1) # batch x src_len
        else:
            attn_h = attn_h.transpose(0, 1).contiguous() # tgt_len x batch x dim
            align_vectors = align_vectors.transpose(0, 1).contiguous() # tgt_len x batch x scr_len

        if self.isCoverage:
            # 只要有 Coverage，肯定是 one_step。
            cov_vec = cov_vec + align_vectors # batch x src_len
            return attn_h, align_vectors, cov_vec
        else:
            return attn_h, align_vectors