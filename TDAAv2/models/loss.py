#coding=utf8
import torch
import torch.nn as nn
import numpy as np
import data.dict as dict
from torch.autograd import Variable

def rank_feas(raw_tgt,feas_list):
    final_num=[]
    for each_feas,each_line in zip(feas_list,raw_tgt):
        line_num=[]
        for spk in each_line:
            line_num.append(each_feas[spk])
        final_num.append(line_num)
    return torch.from_numpy(np.array(final_num))


def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit


def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab


def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores = decoder.compute_score(outputs)
    loss = criterion(scores, targets.view(-1)) + sim_score
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    loss.div(num_total).backward()
    loss = loss.data[0]

    return loss, num_total, num_correct

def ss_loss(config,x_input_map_multi,multi_mask,y_multi_map,loss_multi_func):
    siz=x_input_map_multi.size()
    assert len(siz)==3
    topk=y_multi_map.size()[1]
    x_input_map_multi=torch.unsqueeze(x_input_map_multi,1).expand(siz[0],topk,siz[1],siz[2])

    predict_multi_map=multi_mask*x_input_map_multi
    y_multi_map= Variable(y_multi_map)

    loss_multi_speech=loss_multi_func(predict_multi_map,y_multi_map)

    #各通道和为１的loss部分,应该可以更多的带来差异
    y_sum_map=Variable(torch.ones(config.batch_size,config.mix_speech_len,config.speech_fre)).cuda()
    predict_sum_map=torch.sum(multi_mask,1)
    loss_multi_sum_speech=loss_multi_func(predict_sum_map,y_sum_map)
    print 'loss 1 eval, losssum eval : ',loss_multi_speech.data.cpu().numpy(),loss_multi_sum_speech.data.cpu().numpy()
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print 'evaling multi-abs norm this eval batch:',torch.abs(y_multi_map-predict_multi_map).norm().data.cpu().numpy()
    print 'loss for separation part:',loss_multi_speech.data.cpu().numpy()
    return loss_multi_speech

