#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import data.dict as dict
import models

import numpy as np

class seq2seq(nn.Module):

    def __init__(self, config, input_emb_size, mix_speech_len , tgt_vocab_size, use_cuda, pretrain=None, score_fn=None):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            src_embedding = pretrain['src_emb']
            tgt_embedding = pretrain['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None
        self.encoder = models.rnn_encoder(config, input_emb_size,None, embedding=src_embedding)
        if config.shared_vocab == False:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=tgt_embedding, score_fn=score_fn)
        else:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=self.encoder.embedding, score_fn=score_fn)
        self.use_cuda = use_cuda
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.criterion = models.criterion(tgt_vocab_size, use_cuda)
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        speech_fre=input_emb_size
        num_labels=tgt_vocab_size
        self.ss_model=models.SS(config, speech_fre, mix_speech_len, num_labels)

    def compute_loss(self, hidden_outputs, targets, memory_efficiency):
        if memory_efficiency:
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)

    def separation_loss(self, x_input_map_multi,masks,y_multi_map):
        return models.ss_loss(self.config, x_input_map_multi,masks,y_multi_map,self.loss_for_ss)

    def forward(self, src, src_len, tgt, tgt_len):
        # 感觉这是个把一个batch里的数据按从长到短调整顺序的意思
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        # todo: 这里只要一用排序，tgt那个就出问题，现在的长度都一样，所以没有排序也可以工作，这个得好好研究一下后面
        # src = torch.index_select(src, dim=0, index=indices)
        # tgt = torch.index_select(tgt, dim=0, index=indices)

        contexts, state = self.encoder(src, lengths.data.tolist()) # context是：（max_len,batch_size,hidden_size×2方向）这么大
        if not self.config.global_emb:
            outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
            # 这里的outputs就是没个step输出的隐层向量,大小是len+1,bs,emb（注意是第一个词到 EOS的总共）
            predicted_maps=self.ss_model(src,outputs[:-1,:],tgt[1:-1])
        else:
            outputs, final_state, global_embs = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
            # 这里的outputs就是没个step输出的隐层向量,大小是len+1,bs,emb（注意是第一个词到 EOS的总共）
            predicted_maps=self.ss_model(src,global_embs,tgt[1:-1])

        return outputs, tgt[1:], predicted_maps

    def sample(self, src, src_len):
        # src=src.squeeze()
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS), volatile=True)

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1))
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]
        sample_ids = torch.index_select(sample_ids.data, dim=1, index=ind)
        alignments = torch.index_select(alignments.data, dim=1, index=ind)
        #targets = tgt[1:]

        return sample_ids.t(), alignments.t()


    def beam_sample(self, src, src_len, dict_spk2idx, tgt, beam_size = 1):

        #beam_size = self.config.beam_size
        batch_size = src.size(0)

        # (1) Run the encoder on the src. Done!!!!
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        # _, ind = torch.sort(indices)
        # src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        contexts, encState = self.encoder(src, lengths.data.cpu().numpy()[0])

        #  (1b) Initialize for the decoder.
        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts.data).transpose(0, 1)
        decState = (rvar(encState[0].data), rvar(encState[1].data))
        #decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, dict_spk2idx, n_best=1,
                          cuda=self.use_cuda)
                for __ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.
        
        mask = None
        soft_score = None
        tmp_hiddens=[]
        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))
            if self.config.schmidt and i>0:
                assert len(beam[0].sch_hiddens[-1])==i
                tmp_hiddens=[]
                for xxx in range(i): #每一个sch之前的列表
                    one_len=[]
                    for bm_idx in range(beam_size):
                        for bs_idx in range(batch_size):
                            one_len.append(beam[bs_idx].sch_hiddens[-1][xxx][bm_idx,:])
                    tmp_hiddens.append(var(torch.stack(one_len)))

            # Run one step.
            output, decState, attn ,hidden, emb = self.decoder.sample_one(inp, soft_score, decState, tmp_hiddens, contexts, mask)
            # print "sample after decState:",decState[0].data.cpu().numpy().mean()
            if self.config.schmidt:
                tmp_hiddens+=[hidden]
            if self.config.ct_recu:
                contexts= (1-(attn>0.005).float()).unsqueeze(-1)*contexts
            soft_score = F.softmax(output)
            predicted = output.max(1)[1]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            hidden = unbottle(hidden)
            emb = unbottle(emb)
                # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j], hidden.data[:,j], emb.data[:,j])
                b.beam_update(decState, j) #这个函数更新了原来的decState,只不过不是用return，是直接赋值！
                if self.config.ct_recu:
                    b.beam_update_context(contexts, j) #这个函数更新了原来的decState,只不过不是用return，是直接赋值！
            # print "beam after decState:",decState[0].data.cpu().numpy().mean()

        # (3) Package everything up.
        allHyps, allScores, allAttn, allHiddens, allEmbs= [], [], [], [], []

        ind=range(batch_size)
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn, hiddens, embs = [], [], [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att, hidden, emb = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                hiddens.append(hidden)
                embs.append(emb)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            allHiddens.append(hiddens[0])
            allEmbs.append(embs[0])

        if not self.config.global_emb:
            outputs=Variable(torch.stack(allHiddens,0).transpose(0,1)) # to [decLen, bs, dim]
            predicted_maps=self.ss_model(src,outputs[:-1,:],tgt[1:-1])
        else:
            ss_embs=Variable(torch.stack(allEmbs,0).transpose(0,1)) # to [decLen, bs, dim]
            if not self.config.top1:
                predicted_maps=self.ss_model(src,ss_embs[1:,:],tgt[1:-1])
            else:
                predicted_maps=self.ss_model(src,ss_embs[1:2],tgt[1:2])
        return allHyps, allAttn, allHiddens, predicted_maps