# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import data.dict as dict
import models
# from models.transformer import *
# from figure_hot import relitu_line

import numpy as np


class seq2seq(nn.Module):

    def __init__(self, config, input_emb_size, mix_speech_len, tgt_vocab_size, use_cuda, pretrain=None, score_fn=''):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            src_embedding = pretrain['src_emb']
            tgt_embedding = pretrain['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None
        self.use_cuda = use_cuda
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.criterion = models.criterion(tgt_vocab_size, use_cuda,config.loss)
        self.loss_for_ss = nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()
        self.wav_loss = models.WaveLoss(dBscale=1, nfft=config.FRAME_LENGTH, hop_size=config.FRAME_SHIFT)

        speech_fre = input_emb_size
        num_labels = tgt_vocab_size
        if config.use_tas:
            self.ss_model = models.ConvTasNet(config)
            self.spk_lstm= nn.LSTMCell(self.ss_model.B+self.ss_model.N,self.ss_model.B) # LSTM over the speakers' step.
            if self.config.two_stage:
                self.second_ss_model = models.ConvTasNet_2nd(config)
                for p in self.encoder.parameters():
                    p.requires_grad = False
                for p in self.decoder.parameters():
                    p.requires_grad = False
                for p in self.ss_model.parameters():
                    p.requires_grad = False
        else:
            # self.ss_model = models.SS_att(config, speech_fre, mix_speech_len, num_labels)
            self.ss_model = models.SS(config, speech_fre, mix_speech_len, num_labels)

    def compute_loss(self, hidden_outputs, targets, memory_efficiency):
        if 1:
            return models.cal_performance(hidden_outputs, self.decoder, targets, self.criterion, self.config)

    def separation_loss(self, x_input_map_multi, masks, y_multi_map, Var='NoItem'):
        if not self.config.MLMSE:
            return models.ss_loss(self.config, x_input_map_multi, masks, y_multi_map, self.loss_for_ss,self.wav_loss)
        else:
            return models.ss_loss_MLMSE(self.config, x_input_map_multi, masks, y_multi_map, self.loss_for_ss, Var)

    def silence_loss(self,  est_mask):
        # est: bs, T
        square = est_mask*est_mask #bs,T
        sum = torch.sum(square,1) #bs
        # dist = -10 * torch.log10(sum)
        dist = sum
        return torch.mean(dist)/est_mask.size(0)

    def separation_pit_loss(self, x_input_map_multi, masks, y_multi_map):
        return models.ss_pit_loss(self.config, x_input_map_multi, masks, y_multi_map, self.loss_for_ss,self.wav_loss)

    def separation_tas_loss(self, x_input_wav_multi,predict_wav, y_multi_wav,mix_lengths):
        return models.ss_tas_pit_loss(self.config, x_input_wav_multi,predict_wav, y_multi_wav, mix_lengths,self.loss_for_ss,self.wav_loss)

    def separation_tas_sdr_order_loss(self, x_input_wav_multi,predict_wav, y_multi_wav,mix_lengths):
        return models.ss_tas_loss(self.config, x_input_wav_multi,predict_wav, y_multi_wav, mix_lengths,self.loss_for_ss,self.wav_loss)

    def update_var(self, x_input_map_multi, multi_masks, y_multi_map):
        predict_multi_map = torch.mean(multi_masks * x_input_map_multi, -2)  # 在时间维度上平均
        y_multi_map = torch.mean(Variable(y_multi_map), -2)  # 在时间维度上平均
        loss_vector = (y_multi_map - predict_multi_map).view(-1, self.config.speech_fre).unsqueeze(-1)  # 应该是bs*1*fre
        Var = torch.bmm(loss_vector, loss_vector.transpose(1, 2))
        Var = torch.mean(Var, 0)  # 在batch的维度上平均
        return Var.detach()

    def normalize(self,bs_vec):
        return bs_vec/(torch.max(torch.abs(bs_vec),1)[0]).unsqueeze(1)

    def choose_candidate(self, predicted_wav_this_step, clean_batch_dict, BS):
        # predicted_wav: BS,T , clean_batch_dict: 长度为BS的列表，每个元素是dict，里面是序号对应的向量
        # 每一步选出来一个最近的，然后将它从列表里删除去
        spk_idx_list = []

        cand_wavs_list = []
        for idx in range(BS):
            est_wav = predicted_wav_this_step[idx]  # T
            candidates_dict = clean_batch_dict[idx]  # dict topk,T
            key_max = None
            snr_max = None  # original key and dist
            for key, cand_wav in candidates_dict.items():
                # dist = F.mse_loss(torch.from_numpy(normalization_11(est_wav.data.cpu().numpy())),torch.from_numpy(normalization_11(cand_wav.data.cpu().numpy())))
                # snr = models.cal_si_snr_with_order(cand_wav.view(1,1,-1), est_wav.view(1,1,-1), torch.ones([1]).int().cuda()*cand_wav.shape[-1])
                snr = models.cal_sdr_with_order(cand_wav.view(1,1,-1), est_wav.view(1,1,-1), torch.ones([1]).int().cuda()*cand_wav.shape[-1])
                if snr_max is None:
                    snr_max = snr
                    key_max = key
                else:
                    if snr > snr_max:
                        snr_max = snr
                        key_max = key
            spk_idx_list.append(key_max)
            cand_wavs_list.append(clean_batch_dict[idx][key_max].unsqueeze(0)) # list of 1,T
            clean_batch_dict[idx].pop(key_max)  # 移除该元素

        return cand_wavs_list, spk_idx_list, clean_batch_dict

    def forward(self, src, src_len, tgt, tgt_len, dict_spk2idx,src_original=None,mix_wav=None,clean_wavs=None):
        # 感觉这是个把一个batch里的数据按从长到短调整顺序的意思
        # clean_wavs: topk,BS,T
        # print(src.shape,src_original.shape)
        if src_original is None:
            src_original=src
        if mix_wav is not None:
            mix_wav=mix_wav.transpose(0,1)
        if clean_wavs is not None:
            clean_wavs=clean_wavs.transpose(0,1) # BS,topk,T
            topk=clean_wavs.size(1)
            clean_batch_dict=[]
            for clean_wav in clean_wavs:
                # topk,T
                this_dict={idx:cand for idx,cand in enumerate(clean_wav)}
                clean_batch_dict.append(this_dict)
            # print(clean_batch_dict)



        #  mixture: [BS, T], M is batch
        mixture_w = self.ss_model.encoder(mix_wav) # [BS, N, K],
        mixture_encoder = self.ss_model.separator(mixture_w) # mixture_w: [BS, B, K], where K = (T - L) / (L / 2) + 1 = 2 T / L - 1

        # 注意这里是sep模块之后 还没加conv1x1的情况下

        predicted_map_this_step = Variable(torch.zeros(mix_wav.shape)).to(mix_wav.device) # First step to use all ZEROs
        y_map_this_step = predicted_map_this_step
        condition_last_step = self.ss_model.encoder(y_map_this_step)  # use a conv1d to subsample the original wav to [BS,N,K]
        N, B = self.ss_model.N, self.ss_model.B
        BS, K = mixture_w.shape[0],mixture_w.shape[-1] # new lenght

        predicted_maps = []
        y_maps = []
        spks_list= []
        repeat_time=topk+1 if self.config.add_last_silence else topk
        if self.config.pit_without_tf: # greddy mode w/o Teacher-Forcing
            for step_idx in range(repeat_time):
                cat_condition_this_step= torch.cat((mixture_encoder,condition_last_step),1) #BS,N,K --> BS,B+N,K
                if step_idx==0:
                    # BS,B+N,K --> BS,K,B+N ---> BS*K,B+N --> BS*K,B
                    h_0,c_0=torch.zeros(BS*K, B).to(mix_wav.device),torch.zeros(BS*K, B).to(mix_wav.device)
                    lstm_h,lstm_c = self.spk_lstm(cat_condition_this_step.transpose(1,2).contiguous().view(-1,B+N),(h_0,c_0))
                    del h_0,c_0
                else:
                    # BS,B+N,K --> BS,K,B+N ---> BS*K,B+N --> BS*K,B (lstm_h)
                    lstm_h, lstm_c = self.spk_lstm(cat_condition_this_step.transpose(1,2).contiguous().view(-1,B+N), (lstm_h,lstm_c))

                predicted_map_this_step = self.ss_model.mask_conv1x1(lstm_h.view(-1,K,B).transpose(1,2)) # BS*K,B --> BS,K,B --->BS,B,K--->BS,N,K
                predicted_map_this_step = F.relu(predicted_map_this_step).unsqueeze(1) # BS,1,N,K
                predicted_map_this_step = self.ss_model.decoder(mixture_w, predicted_map_this_step) # BS,1,T
                T_origin =mix_wav.size(-1)
                T_conv = predicted_map_this_step.size(-1)
                predicted_map_this_step = F.pad(predicted_map_this_step, (0, T_origin - T_conv))

                # update the condition
                y_map_this_step = predicted_map_this_step.view(BS,T_origin)
                condition_last_step = self.ss_model.encoder(y_map_this_step)  # use a conv1d to subsample the original wav to [BS,N,K]

                predicted_maps.append(predicted_map_this_step.view(BS,1,T_origin)) # BS,1,T
            predicted_maps = torch.cat(predicted_maps, 1)
            return None, None, None, predicted_maps.transpose(0,1), None

        elif self.config.greddy_tf:
            assert clean_wavs is not None
            for step_idx in range(repeat_time):
                cat_condition_this_step= torch.cat((mixture_encoder,condition_last_step),1) #BS,N,K --> BS,B+N,K
                if step_idx==0:
                    # BS,B+N,K --> BS,K,B+N ---> BS*K,B+N --> BS*K,B
                    h_0,c_0=torch.zeros(BS*K, B).to(mix_wav.device),torch.zeros(BS*K, B).to(mix_wav.device)
                    lstm_h,lstm_c = self.spk_lstm(cat_condition_this_step.transpose(1,2).contiguous().view(-1,B+N),(h_0,c_0))
                    del h_0,c_0
                else:
                    # BS,B+N,K --> BS,K,B+N ---> BS*K,B+N --> BS*K,B (lstm_h)
                    lstm_h, lstm_c = self.spk_lstm(cat_condition_this_step.transpose(1,2).contiguous().view(-1,B+N), (lstm_h,lstm_c))

                predicted_map_this_step = self.ss_model.mask_conv1x1(lstm_h.view(-1,K,B).transpose(1,2)) # BS*K,B --> BS,K,B --->BS,B,K--->BS,N,K
                predicted_map_this_step = F.relu(predicted_map_this_step).unsqueeze(1) # BS,1,N,K
                predicted_map_this_step = self.ss_model.decoder(mixture_w, predicted_map_this_step) # BS,1,T
                T_origin =mix_wav.size(-1)
                T_conv = predicted_map_this_step.size(-1)
                predicted_map_this_step = F.pad(predicted_map_this_step, (0, T_origin - T_conv))
                predicted_maps.append(predicted_map_this_step.view(BS,1,T_origin)) # BS,1,T
                if self.config.add_last_silence and step_idx==repeat_time-1: #如果是最后一个，后面就不用
                    # y_maps.append(torch.zeros(1, BS, T_origin).to(mix_wav.device))  # step个1,BS,T
                    continue

                # update the condition
                y_map_this_step,spk_idx_list_this_step, clean_batch_dict= self.choose_candidate(predicted_map_this_step.view(BS,T_origin),clean_batch_dict,BS)
                print('spk list:',step_idx,spk_idx_list_this_step)
                # print('pred :', step_idx, predicted_map_this_step)
                # print('y wav:',y_map_this_step)
                y_map_this_step = torch.cat(y_map_this_step,0) #BS,T
                # training add some white noise
                # condition_last_step = self.ss_model.encoder(self.normalize(y_map_this_step+0.3*torch.randn(y_map_this_step.shape).to(y_map_this_step.device)))  # use a conv1d to subsample the original wav to [BS,N,K]
                condition_last_step = self.ss_model.encoder(y_map_this_step+0.5*torch.randn(y_map_this_step.shape).to(y_map_this_step.device))  # use a conv1d to subsample the original wav to [BS,N,K]
                # condition_last_step = self.ss_model.encoder(self.normalize(y_map_this_step))  # use a conv1d to subsample the original wav to [BS,N,K]
                # condition_last_step = self.ss_model.encoder(y_map_this_step)  # use a conv1d to subsample the original wav to [BS,N,K]
                y_maps.append(y_map_this_step.view(1,BS,T_origin)) # step个1,BS,T
                spks_list.append(spk_idx_list_this_step)

        predicted_maps=torch.cat(predicted_maps,1)
        y_maps=torch.cat(y_maps,0) # topk,BS,T
        return None, None, torch.from_numpy(np.array(spks_list)).to(y_maps.device), predicted_maps.transpose(0,1), y_maps


        if self.config.two_stage:
            predicted_maps_2nd = self.second_ss_model(mix_wav,predicted_maps) # bs,T   bs,topk,T -->
            return outputs.transpose(0, 1), pred, tgt[1:], predicted_maps.transpose(0, 1),\
                   dec_enc_attn_list, predicted_maps_2nd.transpose( 0, 1)  # n_head*b,topk+1,T

        return outputs.transpose(0,1), pred, tgt[1:], predicted_maps.transpose(0, 1), dec_enc_attn_list  #n_head*b,topk+1,T

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
        # targets = tgt[1:]

        return sample_ids.t(), alignments.t()

    def pit_sample(self, src, src_len, dict_spk2idx, tgt, beam_size=1, src_original=None, mix_wav=None):
        # 感觉这是个把一个batch里的数据按从长到短调整顺序的意思
        # print(src.shape,src_original.shape)
        if src_original is None:
            src_original=src
        src_original=src_original.transpose(0,1) # 确保要bs在第二维
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        # todo: 这里只要一用排序，tgt那个就出问题，现在的长度都一样，所以没有排序也可以工作，这个得好好研究一下后面
        # src = torch.index_select(src, dim=0, index=indices)
        # tgt = torch.index_select(tgt, dim=0, index=indices)

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1) # convert to bs, output_len
        if mix_wav is not None:
            mix_wav=mix_wav.transpose(0,1)
        contexts, *_ = self.encoder(src, lengths.data.tolist(),return_attns=True)  # context是：（batch_size,max_len,hidden_size×2方向）这么大
        if self.config.PIT_training:
            tgt_tmp=tgt.clone()
            tgt_tmp[:,1]=1
            tgt_tmp[:,2]=2
            pred, gold, outputs,embs,dec_slf_attn_list, dec_enc_attn_list= self.decoder(tgt_tmp[:,1:-1], contexts, lengths.data.tolist(),return_attns=True)
        else:
            pred, gold, outputs,embs,dec_slf_attn_list, dec_enc_attn_list= self.decoder(tgt[:,1:-1], contexts, lengths.data.tolist())
        if 0 and self.config.use_emb:
            query=embs[:,1:]
        else:
            query=outputs[:,:-1]
        #outputs: bs,len+1(2+1),emb , embs是类似spk_emb的输入
        tgt = tgt.transpose(0, 1) # convert to output_len(2+2), bs
        # predicted_maps = self.ss_model(src, outputs[:-1], tgt[1:-1], dict_spk2idx)
        predicted_maps = self.ss_model(src_original, query, tgt[1:-1], dict_spk2idx)

        return pred, predicted_maps

    def beam_sample(self, src, src_len, dict_spk2idx, tgt, beam_size=1, src_original=None, mix_wav=None):

        if src_original is None:
            src_original=src
        src_original=src_original.transpose(0,1) # 确保要bs在第二维

        src = src.transpose(0, 1)
        # beam_size = self.config.beam_size
        batch_size = src.size(0)
        if mix_wav is not None:
            mix_wav=mix_wav.transpose(0,1)

        # (1) Run the encoder on the src. Done!!!!
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        # _, ind = torch.sort(indices)
        # src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        contexts, *_ = self.encoder(src, lengths.data.cpu().numpy()[0])
        best_hyps_dict=self.decoder.recognize_beam_greddy(contexts, list(dict_spk2idx.keys()),None)[0]
        print('hyps:',best_hyps_dict['yseq'])

        if self.config.use_emb:
            ss_embs = best_hyps_dict['dec_embs_input'][:,1:]  # to [ bs, decLen(3),dim]
        else:
            ss_embs = best_hyps_dict['dec_hiddens'][:,:-1]  # to [ bs, decLen(3),dim]


        query=ss_embs
        if self.config.use_tas:
            predicted_maps = self.ss_model(mix_wav, query)
        else:
            predicted_maps = self.ss_model(src_original, query, tgt[1:-1], dict_spk2idx)
        return best_hyps_dict['yseq'][1:], predicted_maps.transpose(0,1)

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
        # decState = (rvar(encState[0].data), rvar(encState[1].data))
        # decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, dict_spk2idx, n_best=1,
                            cuda=self.use_cuda)
                for __ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.

        mask = None
        soft_score = None
        tmp_hiddens = []
        tmp_soft_score = []
        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))
            if self.config.schmidt and i > 0:
                assert len(beam[0].sch_hiddens[-1]) == i
                tmp_hiddens = []
                for xxx in range(i):  # 每一个sch之前的列表
                    one_len = []
                    for bm_idx in range(beam_size):
                        for bs_idx in range(batch_size):
                            one_len.append(beam[bs_idx].sch_hiddens[-1][xxx][bm_idx, :])
                    tmp_hiddens.append(var(torch.stack(one_len)))

            # Run one step.
            output, decState, attn, hidden, emb = self.decoder.sample_one(inp, soft_score, tmp_hiddens,
                                                                          contexts.transpose(0,1), mask)
            # print "sample after decState:",decState[0].data.cpu().numpy().mean()
            if self.config.schmidt:
                tmp_hiddens += [hidden]
            if self.config.ct_recu:
                contexts = (1 - (attn > 0.03).float()).unsqueeze(-1) * contexts
            soft_score = F.softmax(output)
            if self.config.tmp_score:
                tmp_soft_score += [soft_score]
                if i == 1:
                    kl_loss = np.array([])
                    for kk in range(self.config.beam_size):
                        kl_loss = np.append(kl_loss, F.kl_div(soft_score[kk], tmp_soft_score[0][kk].data).data[0])
                    kl_loss = Variable(torch.from_numpy(kl_loss).float().cuda().unsqueeze(-1))
            predicted = output.max(1)[1]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if self.config.tmp_score and i == 1:
                output = unbottle(self.log_softmax(output) + self.config.tmp_score * kl_loss)
            else:
                output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            hidden = unbottle(hidden)
            emb = unbottle(emb)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j], hidden.data[:, j], emb.data[:, j])
                # b.beam_update(decState, j)  # 这个函数更新了原来的decState,只不过不是用return，是直接赋值！
                # print('pre root',b.prevKs)
                # print('next root',b.nextYs)
                # print('score',b.scores)
                if self.config.ct_recu:
                    b.beam_update_context(contexts, j)  # 这个函数更新了原来的decState,只不过不是用return，是直接赋值！
            # print "beam after decState:",decState[0].data.cpu().numpy().mean()

        # (3) Package everything up.
        allHyps, allScores, allAttn, allHiddens, allEmbs = [], [], [], [], []

        ind = list(range(batch_size))
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn, hiddens, embs = [], [], [], []
            pred = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att, hidden, emb = b.getHyp(times, k)
                # if self.config.relitu:
                #     relitu_line(626, 1, att[0].cpu().numpy())
                #     relitu_line(626, 1, att[1].cpu().numpy())
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                hiddens.append(hidden)
                embs.append(emb)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            allHiddens.append(hiddens[0])
            allEmbs.append(embs[0])
        print('allHyps:\n',allHyps)

        # from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
        # print(cosine_distances(allEmbs[0].data.cpu().numpy()))
        # print(cosine_distances(allHiddens[0].data.cpu().numpy()))

        if not self.config.global_emb:
            outputs = Variable(torch.stack(allHiddens, 0).transpose(0, 1))  # to [decLen, bs, dim]
            if not self.config.hidden_mix:
                predicted_maps = self.ss_model(src, outputs[:-1, :], tgt[1:-1])
            else:
                ss_embs = Variable(torch.stack(allEmbs, 0).transpose(0, 1))  # to [decLen, bs, dim]
                mix = torch.cat((outputs[:-1, :], ss_embs[1:]), dim=2)
                predicted_maps = self.ss_model(src, mix, tgt[1:-1])
            if self.config.top1:
                predicted_maps = predicted_maps[:, 0].unsqueeze(1)
        else:
            # allEmbs=[j[1:self.config.MAX_MIX] for j in allEmbs]
            ss_embs = Variable(torch.stack(allEmbs, 0).transpose(0, 1))  # to [decLen, bs, dim]
            if self.config.use_tas:
                if self.config.global_hidden:
                    ss_hidden = Variable(torch.stack(allHiddens, 0).transpose(0, 1))  # to [decLen, bs, dim]
                    # predicted_maps = self.ss_model(mix_wav,ss_hidden[:-1, :]) #正确
                    predicted_maps = self.ss_model(mix_wav,ss_hidden[1:, :].transpose(0,1)) #错位
                else:
                    predicted_maps = self.ss_model(mix_wav,ss_embs[1:].transpose(0,1))
            elif self.config.global_hidden:
                ss_hidden = Variable(torch.stack(allHiddens, 0).transpose(0, 1))  # to [decLen, bs, dim]
                print((ss_hidden.shape))

                predicted_maps = self.ss_model(src,ss_hidden[1:, :], tgt[1:-1], dict_spk2idx)
            elif self.config.top1:
                predicted_maps = self.ss_model(src, ss_embs[1:2], tgt[1:2])
            else:
                predicted_maps = self.ss_model(src, ss_embs[1:, :], tgt[1:-1], dict_spk2idx)
                # predicted_maps = self.ss_model(src, ss_embs, tgt[1:2])
        if self.config.top1:
            predicted_maps = predicted_maps[:,0:1] #bs,1,len
        return allHyps, allAttn, allHiddens, predicted_maps  # .transpose(0,1)


if __name__ == '__main__':
    model=seq2seq()
