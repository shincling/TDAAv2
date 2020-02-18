import torch.nn as nn
import numpy as np

from models.transformer_utils import(IGNORE_ID, get_attn_key_pad_mask, get_attn_pad_mask,
                   get_non_pad_mask, get_subsequent_mask, pad_list)

# Attention Class"
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn # (n*b) x lq x lq


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


# Another implementation
class PositionwiseFeedForwardUseConv(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output




class TransEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, config, d_input, n_layers=2, n_head=4, d_k=64, d_v=64,
                 d_model=512, d_inner=2048, dropout=0.1, pe_maxlen=5000):
        super(TransEncoder, self).__init__()
        # parameters
        self.config = config
        self.d_input = d_input
        self.n_layers = config.trans_n_layers
        self.n_head = config.trans_n_head
        self.d_k = config.trans_d_k
        self.d_v = config.trans_d_v
        self.d_model = config.trans_d_model
        self.d_inner = config.trans_d_inner
        self.dropout_rate = config.trans_dropout
        self.pe_maxlen = config.trans_pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn] # n_layer个 (head*bs) x lq x dk

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn #(n*b) x lq x lq

class TransDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, config, sos_id, eos_id,
            n_tgt_vocab, d_word_vec=512,
            n_layers=1, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1,
            tgt_emb_prj_weight_sharing=False,
            pe_maxlen=5000):
        super(TransDecoder, self).__init__()
        # parameters
        self.config = config
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                     seq_q=ys_in_pad,
                                                     pad_idx=self.eos_id)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        output_length = ys_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length)

        # Forward
        dec_output_input = self.dropout(self.tgt_word_emb(ys_in_pad) * self.x_logit_scale +
                                  self.positional_encoding(ys_in_pad))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_input, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if 1 or return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        # dec_output: bs,dec_len,512
        seq_logit = self.tgt_word_prj(dec_output)

        # Return
        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_output, dec_output_input, dec_slf_attn_list, dec_enc_attn_list[-1].view(self.n_head,-1,3,751)
        return pred, gold, dec_output, dec_output_input

    def compute_score(self, hiddens,targets):
        scores = self.tgt_word_prj(hiddens)
        return scores

    def sample(self, input, init_state, contexts):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len
        soft_score = None
        mask = None
        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], soft_score, state, contexts, mask)
            if self.config.global_emb:
                soft_score = F.softmax(output)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, soft_score, tmp_hiddens, contexts, mask):
        input = input.unsqueeze(0)
        non_pad_mask = torch.ones_like(input).float().unsqueeze(-1)  # 1xix1
        slf_attn_mask = get_subsequent_mask(input)
        dec_output = self.dropout(
            self.tgt_word_emb(input) * self.x_logit_scale +
            self.positional_encoding(input))

        for dec_layer in self.layer_stack:
            dec_output, _, _ = dec_layer(
                dec_output, contexts,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=None)

        output = self.compute_score(dec_output,targets=None)
        # print(output)
        if 1:
            if mask is not None:
                output = output.scatter_(1, mask, -9999999999)
        # return output, state, attn_weigths, hidden, emb
        return output, dec_output, dec_output, dec_output, dec_output

    def recognize_beam(self, encoder_outputs, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam
        Returns:
            nbest_hyps:
        """
        # search params
        beam = 5
        nbest = 1
        # if args.decode_max_len == 0:
        #     maxlen = encoder_outputs.size(0)
        # else:
        #     maxlen = args.decode_max_len
        maxlen = 5

        # encoder_outputs = encoder_outputs.unsqueeze(0)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i

                # -- Prepare masks
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                slf_attn_mask = get_subsequent_mask(ys)

                # -- Forward
                dec_output_input = self.dropout(
                    self.tgt_word_emb(ys) * self.x_logit_scale +
                    self.positional_encoding(ys))

                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(
                        dec_output_input, encoder_outputs,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=None)

                seq_logit = self.tgt_word_prj(dec_output[:, -1])

                local_scores = F.log_softmax(seq_logit, dim=1)
                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1+ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    new_hyp['dec_hiddens']=dec_output
                    new_hyp['dec_embs_input']=dec_output_input
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)


                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()], dim=1)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                print('remeined hypothes: ' + str(len(hyps)))
            else:
                print('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                print('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][0, 1:]]), hyp['score'])
        # end for i in range(maxlen)

        tmp_list=[]
        for hyp in ended_hyps:
            if len(set(list(hyp['yseq'][0].data.cpu().numpy())))==len(hyp['yseq'][0]):
                tmp_list.append(hyp)
            else:
                print('Repeated Predtion:', hyp['yseq'])
        ended_hyps = tmp_list

        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            :min(len(ended_hyps), nbest)]
        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        return nbest_hyps

    def recognize_pit(self, padded_input, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                     seq_q=ys_in_pad,
                                                     pad_idx=self.eos_id)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        output_length = ys_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length)

        # Forward
        dec_output_input = self.dropout(self.tgt_word_emb(ys_in_pad) * self.x_logit_scale +
                                        self.positional_encoding(ys_in_pad))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_input, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if 1 or return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        # dec_output: bs,dec_len,512
        seq_logit = self.tgt_word_prj(dec_output)

        # Return
        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_output, dec_output_input, dec_slf_attn_list, dec_enc_attn_list[-1].view(self.n_head,-1,3,751)
        return pred, gold, dec_output, dec_output_input

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


# the total  model
class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        return pred, gold

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model, LFR_m, LFR_n = cls.load_model_from_package(package)
        return model, LFR_m, LFR_n

    @classmethod
    def load_model_from_package(cls, package):
        encoder = TransEncoder(package['d_input'],
                          package['n_layers_enc'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          pe_maxlen=package['pe_maxlen'])
        decoder = TransDecoder(package['sos_id'],
                          package['eos_id'],
                          package['vocab_size'],
                          package['d_word_vec'],
                          package['n_layers_dec'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          tgt_emb_prj_weight_sharing=package['tgt_emb_prj_weight_sharing'],
                          pe_maxlen=package['pe_maxlen'],
                          )
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        LFR_m, LFR_n = package['LFR_m'], package['LFR_n']
        return model, LFR_m, LFR_n

    @staticmethod
    def serialize(model, optimizer, epoch, LFR_m, LFR_n, tr_loss=None, cv_loss=None):
        package = {
            # Low Frame Rate Feature
            'LFR_m': LFR_m,
            'LFR_n': LFR_n,
            # encoder
            'd_input': model.encoder.d_input,
            'n_layers_enc': model.encoder.n_layers,
            'n_head': model.encoder.n_head,
            'd_k': model.encoder.d_k,
            'd_v': model.encoder.d_v,
            'd_model': model.encoder.d_model,
            'd_inner': model.encoder.d_inner,
            'dropout': model.encoder.dropout_rate,
            'pe_maxlen': model.encoder.pe_maxlen,
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_sharing': model.decoder.tgt_emb_prj_weight_sharing,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package