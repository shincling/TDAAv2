#coding=utf8
import torch
import torch.nn as nn
import numpy as np
import data.dict as dict
from torch.autograd import Variable
import torch.nn.functional as F

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
    loss=loss.div(num_total)
    # loss = loss.data[0]

    return loss, num_total, num_correct

def ss_loss(config,x_input_map_multi,multi_mask,y_multi_map,loss_multi_func):
    predict_multi_map=multi_mask*x_input_map_multi
    # predict_multi_map=Variable(y_multi_map)
    y_multi_map= Variable(y_multi_map)

    loss_multi_speech=loss_multi_func(predict_multi_map,y_multi_map)

    #各通道和为１的loss部分,应该可以更多的带来差异
    y_sum_map=Variable(torch.ones(config.batch_size,config.mix_speech_len,config.speech_fre)).cuda()
    predict_sum_map=torch.sum(multi_mask,1)
    loss_multi_sum_speech=loss_multi_func(predict_sum_map,y_sum_map)
    print 'loss 1 eval, losssum eval : ',loss_multi_speech.data.cpu().numpy(),loss_multi_sum_speech.data.cpu().numpy()
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print 'evaling multi-abs norm this eval batch:',torch.abs(y_multi_map-predict_multi_map).norm().data.cpu().numpy()
    # loss_multi_speech=loss_multi_speech+3*loss_multi_sum_speech
    print 'loss for whole separation part:',loss_multi_speech.data.cpu().numpy()
    return loss_multi_speech

def ss_loss_MLMSE(config,x_input_map_multi,multi_mask,y_multi_map,loss_multi_func,Var):
    try:
        if Var == None:
            Var=Variable(torch.eye(config.speech_fre,config.speech_fre).cuda(),requires_grad=0) #初始化的是单位矩阵
            print 'Set Var to:',Var
    except:
        pass
    assert Var.size()==(config.speech_fre,config.speech_fre)

    predict_multi_map=torch.mean(multi_mask*x_input_map_multi,-2) #在时间维度上平均
    # predict_multi_map=Variable(y_multi_map)
    y_multi_map= torch.mean(Variable(y_multi_map),-2) #在时间维度上平均

    loss_vector=(y_multi_map-predict_multi_map).view(-1,config.speech_fre).unsqueeze(1) #应该是bs*1*fre

    Var_inverse=torch.inverse(Var)
    Var_inverse=Var_inverse.unsqueeze(0).expand(loss_vector.size()[0], config.speech_fre, config.speech_fre) #扩展成batch的形式
    loss_multi_speech=torch.bmm(torch.bmm(loss_vector,Var_inverse),loss_vector.transpose(1,2))
    loss_multi_speech=torch.mean(loss_multi_speech,0)

    #各通道和为１的loss部分,应该可以更多的带来差异
    y_sum_map=Variable(torch.ones(config.batch_size,config.mix_speech_len,config.speech_fre)).cuda()
    predict_sum_map=torch.sum(multi_mask,1)
    loss_multi_sum_speech=loss_multi_func(predict_sum_map,y_sum_map)
    print 'loss 1 eval, losssum eval : ',loss_multi_speech.data.cpu().numpy(),loss_multi_sum_speech.data.cpu().numpy()
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print 'evaling multi-abs norm this eval batch:',torch.abs(y_multi_map-predict_multi_map).norm().data.cpu().numpy()
    # loss_multi_speech=loss_multi_speech+3*loss_multi_sum_speech
    print 'loss for whole separation part:',loss_multi_speech.data.cpu().numpy()
    # return F.relu(loss_multi_speech)
    return loss_multi_speech

def dis_loss(config,top_k_num,dis_model,x_input_map_multi,multi_mask,y_multi_map,loss_multi_func):
    predict_multi_map=multi_mask*x_input_map_multi
    y_multi_map= Variable(y_multi_map).cuda()
    score_true=dis_model(y_multi_map)
    score_false=dis_model(predict_multi_map)
    acc_true=torch.sum(score_true>0.5).data.cpu().numpy()/float(score_true.size()[0])
    acc_false=torch.sum(score_false<0.5).data.cpu().numpy()/float(score_true.size()[0])
    acc_dis=(acc_false+acc_true)/2
    print 'acc for dis:(ture,false,aver)',acc_true,acc_false,acc_dis

    loss_dis_true=loss_multi_func(score_true,Variable(torch.ones(config.batch_size*top_k_num,1)).cuda())
    loss_dis_false=loss_multi_func(score_false,Variable(torch.zeros(config.batch_size*top_k_num,1)).cuda())
    loss_dis=loss_dis_true+loss_dis_false
    print 'loss for dis:(ture,false)',loss_dis_true.data.cpu().numpy(),loss_dis_false.data.cpu().numpy()
    return loss_dis
