# coding=utf8
import os
import argparse
import time
import json
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import lera
from tensorboardX import SummaryWriter

import models
import data.utils as utils
from optims import Optim
import lr_scheduler as L
# from predata_CN import prepare_data  # 数据准备的模块
from predata_fromList_123 import prepare_data  # 数据准备的模块
# from predata_fromList_dynamic import prepare_data  # 数据准备的模块
# from predata_CN_aim import prepare_data as prepare_data_aim # 测试阶段随便一段语音的数据准备脚本
import bss_test  # 语音分离性能评测脚本


# config
parser = argparse.ArgumentParser(description='train_WSJ0.py')

parser.add_argument('-config', default='config_WSJ0.yaml', type=str,
                    help="config file")
# parser.add_argument('-gpus', default=range(4), nargs='+', type=int,
parser.add_argument('-gpus', default=[2,3], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
# parser.add_argument('-restore', default='TDAAv3_PIT_30001.pt', type=str,
parser.add_argument('-restore', default='Tranformer_PIT_30001.pt', type=str,
# parser.add_argument('-restore', default='data/data/log/2020-02-07-05:01:34/Transformer_PIT_58001.pt', type=str,
# parser.add_argument('-restore', default=None, type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-notrain', default=0, type=bool,
                    help="train or not")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
parser.add_argument('-memory', default=False, type=bool,
                    help="memory efficiency")
# parser.add_argument('-score_fc', default='arc_margin', type=str,
parser.add_argument('-score_fc', default='', type=str,
                    help="memory efficiency")
parser.add_argument('-label_dict_file', default='./data/data/rcv1.json', type=str,
                    help="label_dict")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore:
    print(('loading checkpoint...\n', opt.restore))
    checkpoints = torch.load(opt.restore,map_location={'cuda:6':'cuda:0'})
    # checkpoints = torch.load(opt.restore)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
# use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)

# load the global statistic of the data
print('loading data...\n')
start_time = time.time()

spk_global_gen = prepare_data(mode='global', train_or_test='train')  # 数据中的一些统计参数的读取
# global_para = spk_global_gen.next()
global_para = next(spk_global_gen)
print(global_para)

spk_all_list = global_para['all_spk']  # 所有说话人的列表
dict_spk2idx = global_para['dict_spk_to_idx']
dict_idx2spk = global_para['dict_idx_to_spk']
speech_fre = global_para['num_fre']  # 语音频率总数
total_frames = global_para['num_frames']  # 语音长度
spk_num_total = global_para['total_spk_num']  # 总计说话人数目
batch_total = global_para['total_batch_num']  # 一个epoch里多少个batch

config.speech_fre = speech_fre
mix_speech_len = total_frames
config.mix_speech_len = total_frames
num_labels = len(spk_all_list)
del spk_global_gen
print(('loading the global setting cost: %.3f' % (time.time() - start_time)))

# model
print('building model...\n')
# 调了model.seq2seq 并且运行了最后这个括号里的五个参数的方法。(初始化了一个对象也就是）
model = getattr(models, opt.model)(config, speech_fre, mix_speech_len, num_labels, use_cuda, None, opt.score_fc)

if config.use_center_loss:
    center_loss=models.CenterLoss(num_classes=num_labels,feat_dim=config.SPK_EMB_SIZE,use_gpu=True)
    print(('Here we use center loss:',center_loss))

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if 0 and opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

if config.use_center_loss:
    optim.set_parameters(list(model.parameters())+list(center_loss.parameters()))
else:
    optim.set_parameters(list(model.parameters()))

if 0: # 如果只更新后面的部分
    for k,v in model.encoder.named_parameters():
        v.requires_grad=False
    for k,v in model.decoder.named_parameters():
        v.requires_grad=False
if config.schedule:
    # scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    scheduler = L.StepLR(optim.optimizer, step_size=15, gamma=0.3)

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

# logging modeule
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
print(('log_path:',log_path))

writer=SummaryWriter(log_path)

logging = utils.logging(log_path + 'log.txt')  # 单独写一个logging的函数，直接调用，既print，又记录到Log文件里。
logging_csv = utils.logging_csv(log_path + 'record.csv')
for k, v in list(config.items()):
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model) + "\n\n")

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

total_loss, start_time = 0, time.time()
total_loss_sgm, total_loss_ss = 0, 0
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(list(zip(config.metric, scores)))
best_SDR = 0.0

# train
global_par_dict={
    'title': str('Transformer PIT'),
    'updates': updates,
    'batch_size': config.batch_size,
    'log path': str(log_path),
    'loss':str(config.loss),
    'score fnc': str(opt.score_fc),
    'trans_n_layers': config.trans_n_layers,
    'trans_n_head':  config.trans_n_head,
    'trans_d_k': config.trans_d_k,
    'trans_d_v':  config.trans_d_v,
    'trans_d_model': config.trans_d_model,
    'trans_d_inner': config.trans_d_inner,
    'trans_dropout': config.trans_dropout,
}
lera.log_hyperparams(global_par_dict)
for item in list(global_par_dict.keys()):
    writer.add_text( item, str(global_par_dict[item]))


def train(epoch):
    global e, updates, total_loss, start_time, report_total,report_correct, total_loss_sgm, total_loss_ss
    e = epoch
    model.train()
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])

    if updates<=config.warmup: #如果不在warm阶段就正常规划
       pass
    elif config.schedule and scheduler.get_lr()[0]>4e-5:
        scheduler.step()
        print(("Decaying learning rate to %g" % scheduler.get_lr()[0],updates))
        lera.log({
            'lr': [group['lr'] for group in optim.optimizer.param_groups][0],
        })

    if opt.model == 'gated':
        model.current_epoch = epoch


    # train_data_gen = prepare_data('once', 'train')
    train_data_gen = prepare_data('once', 'valid')
    while True:
        if updates <= config.warmup:  # 如果在warm就开始warmup
            tmp_lr =  config.learning_rate * min(max(updates,1)** (-0.5),
                                             max(updates,1) * (config.warmup ** (-1.5)))
            for param_group in optim.optimizer.param_groups:
                param_group['lr'] = tmp_lr
            scheduler.base_lrs=list([group['lr'] for group in optim.optimizer.param_groups])
            if updates%100==0: #记录一下
                print(updates)
                print("Warmup learning rate to %g" % tmp_lr)
                lera.log({
                    'lr': [group['lr'] for group in optim.optimizer.param_groups][0],
                })

        train_data = next(train_data_gen)
        if train_data == False:
            print(('SDR_aver_epoch:', SDR_SUM.mean()))
            print(('SDRi_aver_epoch:', SDRi_SUM.mean()))
            break  # 如果这个epoch的生成器没有数据了，直接进入下一个epoch

        src = Variable(torch.from_numpy(train_data['mix_feas']))
        # raw_tgt = [spk.keys() for spk in train_data['multi_spk_fea_list']]
        # raw_tgt = [sorted(spk.keys()) for spk in train_data['multi_spk_fea_list']]
        raw_tgt=train_data['batch_order']
        feas_tgt = models.rank_feas(raw_tgt, train_data['multi_spk_fea_list'])  # 这里是目标的图谱,bs*Topk,len,fre

        # 要保证底下这几个都是longTensor(长整数）
        tgt_max_len = config.MAX_MIX + 2  # with bos and eos.
        tgt = Variable(torch.from_numpy(np.array(
            [[0] + [dict_spk2idx[spk] for spk in spks] + (tgt_max_len - len(spks) - 1) * [dict_spk2idx['<EOS>']] for
             spks in raw_tgt], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。
        # tgt = Variable(torch.from_numpy(np.array([[0,1,2,102] for __ in range(config.batch_size)], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。
        src_len = Variable(torch.LongTensor(config.batch_size).zero_() + mix_speech_len).unsqueeze(0)
        tgt_len = Variable(
            torch.LongTensor([len(one_spk) for one_spk in train_data['multi_spk_fea_list']])).unsqueeze(0)
        if config.WFM:
            siz = src.size()  # bs,T,F
            assert len(siz) == 3
            # topk_max = config.MAX_MIX  # 最多可能的topk个数
            topk_max = 2  # 最多可能的topk个数
            x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous().view(-1, siz[1], siz[ 2])  # bs,topk,T,F
            feas_tgt_tmp = feas_tgt.view(siz[0], -1, siz[1], siz[2])

            feas_tgt_square = feas_tgt_tmp * feas_tgt_tmp
            feas_tgt_sum_square = torch.sum(feas_tgt_square, dim=1, keepdim=True).expand(siz[0], topk_max, siz[1], siz[2])
            WFM_mask = feas_tgt_square / (feas_tgt_sum_square + 1e-15)
            feas_tgt = x_input_map_multi.view(siz[0], -1, siz[1], siz[2]).data * WFM_mask  # bs,topk,T,F
            feas_tgt = feas_tgt.view(-1, siz[1], siz[2])  # bs*topk,T,F
            WFM_mask = WFM_mask.cuda()
            del x_input_map_multi

        elif config.PSM:
            siz = src.size()  # bs,T,F
            assert len(siz) == 3
            # topk_max = config.MAX_MIX  # 最多可能的topk个数
            topk_max = 2  # 最多可能的topk个数
            x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous()  # bs,topk,T,F
            feas_tgt_tmp = feas_tgt.view(siz[0], -1, siz[1], siz[2])

            IRM=feas_tgt_tmp/(x_input_map_multi+1e-15)

            angle_tgt=models.rank_feas(raw_tgt, train_data['multi_spk_angle_list']).view(siz[0],-1,siz[1],siz[2])
            angle_mix=Variable(torch.from_numpy(np.array(train_data['mix_angle']))).unsqueeze(1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous()
            ang=np.cos(angle_mix-angle_tgt)
            ang=np.clip(ang,0,None)

            # feas_tgt = x_input_map_multi *np.clip(IRM.numpy()*ang,0,1) # bs,topk,T,F
            # feas_tgt = x_input_map_multi *IRM*ang # bs,topk,T,F
            feas_tgt = feas_tgt.view(siz[0],-1,siz[1],siz[2])*ang # bs,topk,T,F
            feas_tgt = feas_tgt.view(-1, siz[1], siz[2])  # bs*topk,T,F
            del x_input_map_multi

        elif config.frame_mask:
            siz = src.size()  # bs,T,F
            assert len(siz) == 3
            # topk_max = config.MAX_MIX  # 最多可能的topk个数
            topk_max = 2  # 最多可能的topk个数
            x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous()  # bs,topk,T,F
            feas_tgt_tmp = feas_tgt.view(siz[0], -1, siz[1], siz[2])

            feas_tgt_time=torch.sum(feas_tgt_tmp,3).transpose(1,2) #bs,T,topk
            for v1 in feas_tgt_time:
                for v2 in v1:
                    if v2[0]>v2[1]:
                        v2[0]=1
                        v2[1]=0
                    else:
                        v2[0]=0
                        v2[1]=1
            frame_mask=feas_tgt_time.transpose(1,2).unsqueeze(-1) #bs,topk,t,1
            feas_tgt=x_input_map_multi*frame_mask
            feas_tgt = feas_tgt.view(-1, siz[1], siz[2])  # bs*topk,T,F


        if use_cuda:
            src = src.cuda().transpose(0, 1)
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()

        model.zero_grad()
        if config.use_center_loss:
            center_loss.zero_grad()

        # aim_list 就是找到有正经说话人的地方的标号
        aim_list = (tgt[1:-1].transpose(0, 1).contiguous().view(-1) != dict_spk2idx['<EOS>']).nonzero().squeeze()
        aim_list = aim_list.data.cpu().numpy()

        multi_mask, enc_attn_list = model(src, src_len, tgt, tgt_len,
                                             dict_spk2idx)  # 这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        print('mask size:', multi_mask.size()) # topk,bs,T,F
        # print('mask:', multi_mask[0,0,:3:3]) # topk,bs,T,F
        # writer.add_histogram('global gamma',gamma, updates)


        src = src.transpose(0, 1)
        # expand the raw mixed-features to topk_max channel.
        siz = src.size()
        assert len(siz) == 3
        topk_max = config.MAX_MIX  # 最多可能的topk个数
        x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous()#.view(-1, siz[1], siz[2])
        # x_input_map_multi = x_input_map_multi[aim_list]
        # x_input_map_multi = x_input_map_multi.transpose(0, 1) #topk,bs,T,F
        multi_mask = multi_mask.transpose(0, 1)
        # if config.WFM:
        #     feas_tgt = x_input_map_multi.data * WFM_mask

        # 注意,bs是第二维
        assert multi_mask.shape == x_input_map_multi.shape
        assert multi_mask.size(0) == config.batch_size

        if 1 and len(opt.gpus) > 1: #先ss获取Perm
            ss_loss, best_pmt = model.module.separation_pit_loss(x_input_map_multi, multi_mask, feas_tgt)
        else:
            ss_loss, best_pmt = model.separation_pit_loss(x_input_map_multi, multi_mask, feas_tgt)
        print('loss for SS,this batch:', ss_loss.cpu().item())
        print('best perms for this batch:', best_pmt)
        writer.add_scalars('scalar/loss',{'ss_loss':ss_loss.cpu().item()},updates)

        loss = ss_loss
        loss.backward()

        total_loss_ss += ss_loss.cpu().item()
        lera.log({
            'ss_loss': ss_loss.cpu().item(),
        })

        if updates>3 and updates % config.eval_interval in [0, 1, 2, 3, 4, 5]:
            assert multi_mask.shape==x_input_map_multi.shape
            assert multi_mask.size(0)==config.batch_size
            predicted_maps = (multi_mask * x_input_map_multi).view(siz[0]*topk_max,siz[1],siz[2])

            # predicted_maps=Variable(feas_tgt)
            utils.bss_eval(config, predicted_maps, train_data['multi_spk_fea_list'], raw_tgt, train_data, dst=log_path+'batch_output/')
            # utils.bss_eval2(config, predicted_maps, train_data['multi_spk_fea_list'], raw_tgt, train_data, dst='batch_output1')
            del predicted_maps, multi_mask, x_input_map_multi
            sdr_aver_batch, sdri_aver_batch=  bss_test.cal(log_path+'batch_output/')
            lera.log({'SDR sample': sdr_aver_batch})
            lera.log({'SDRi sample': sdri_aver_batch})
            writer.add_scalars('scalar/loss',{'SDR_sample':sdr_aver_batch,'SDRi_sample':sdri_aver_batch},updates)
            SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
            SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
            print(('SDR_aver_now:', SDR_SUM.mean()))
            print(('SDRi_aver_now:', SDRi_SUM.mean()))

        total_loss += loss.cpu().item()
        optim.step()

        updates += 1
        if updates % 30 == 0:
            logging(
                "time: %6.3f, epoch: %3d, updates: %8d, train loss this batch: %6.3f,ss loss: %6.6f\n"
                % (time.time() - start_time, epoch, updates, loss , total_loss_ss / 30.0))
            total_loss_sgm, total_loss_ss = 0, 0

        # continue

        if 0 and updates % config.eval_interval == 0 and epoch > 3: #建议至少跑几个epoch再进行测试，否则模型还没学到东西，会有很多问题。
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f\n"
                    % (time.time() - start_time, epoch, updates, total_loss / report_total))
            print(('evaluating after %d updates...\r' % updates))
            original_bs=config.batch_size
            score = eval(epoch) # eval的时候batch_size会变成1
            # print 'Orignal bs:',original_bs
            config.batch_size=original_bs
            # print 'Now bs:',config.batch_size
            for metric in config.metric:
                scores[metric].append(score[metric])
                lera.log({
                    'sgm_micro_f1': score[metric],
                })
                if metric == 'micro_f1' and score[metric] >= max(scores[metric]):
                    save_model(log_path + 'best_' + metric + '_checkpoint.pt')
                if metric == 'hamming_loss' and score[metric] <= min(scores[metric]):
                    save_model(log_path + 'best_' + metric + '_checkpoint.pt')

            model.train()
            total_loss = 0
            start_time = 0
            report_total = 0
            report_correct = 0

        if 1 and updates % config.save_interval == 1:
            save_model(log_path + 'Transformer_PIT_{}.pt'.format(updates))


def eval(epoch):
    # config.batch_size=1
    model.eval()
    # print '\n\n测试的时候请设置config里的batch_size为1！！！please set the batch_size as 1'
    reference, candidate, source, alignments = [], [], [], []
    e = epoch
    test_or_valid = 'test'
    # test_or_valid = 'valid'
    # test_or_valid = 'train'
    print(('Test or valid:', test_or_valid))
    eval_data_gen = prepare_data('once', test_or_valid, config.MIN_MIX, config.MAX_MIX)
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])
    batch_idx = 0
    global best_SDR, Var
    # for iii in range(2000):
    while True:
        print(('-' * 30))
        eval_data = next(eval_data_gen)
        if eval_data == False:
            print(('SDR_aver_eval_epoch:', SDR_SUM.mean()))
            print(('SDRi_aver_eval_epoch:', SDRi_SUM.mean()))
            break  # 如果这个epoch的生成器没有数据了，直接进入下一个epoch
        src = Variable(torch.from_numpy(eval_data['mix_feas']))

        # raw_tgt = [sorted(spk.keys()) for spk in eval_data['multi_spk_fea_list']]
        raw_tgt= eval_data['batch_order']
        feas_tgt = models.rank_feas(raw_tgt, eval_data['multi_spk_fea_list'])  # 这里是目标的图谱

        top_k = len(raw_tgt[0])
        # 要保证底下这几个都是longTensor(长整数）
        # tgt = Variable(torch.from_numpy(np.array([[0]+[dict_spk2idx[spk] for spk in spks]+[dict_spk2idx['<EOS>']] for spks in raw_tgt],dtype=np.int))).transpose(0,1) #转换成数字，然后前后加开始和结束符号。
        tgt = Variable(torch.from_numpy(np.array([[0,1,2,102] for __ in range(config.batch_size)], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。

        src_len = Variable(torch.LongTensor(config.batch_size).zero_() + mix_speech_len).unsqueeze(0)
        tgt_len = Variable(torch.LongTensor([len(one_spk) for one_spk in eval_data['multi_spk_fea_list']])).unsqueeze(0)
        # tgt_len = Variable(torch.LongTensor(config.batch_size).zero_()+len(eval_data['multi_spk_fea_list'][0])).unsqueeze(0)
        if config.WFM:
            siz = src.size()  # bs,T,F
            assert len(siz) == 3
            # topk_max = config.MAX_MIX  # 最多可能的topk个数
            topk_max = 2  # 最多可能的topk个数
            x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous().view(-1, siz[1], siz[ 2])  # bs,topk,T,F
            feas_tgt_tmp = feas_tgt.view(siz[0], -1, siz[1], siz[2])

            feas_tgt_square = feas_tgt_tmp * feas_tgt_tmp
            feas_tgt_sum_square = torch.sum(feas_tgt_square, dim=1, keepdim=True).expand(siz[0], topk_max, siz[1], siz[2])
            WFM_mask = feas_tgt_square / (feas_tgt_sum_square + 1e-15)
            feas_tgt = x_input_map_multi.view(siz[0], -1, siz[1], siz[2]).data * WFM_mask  # bs,topk,T,F
            feas_tgt = feas_tgt.view(-1, siz[1], siz[2])  # bs*topk,T,F
            WFM_mask = WFM_mask.cuda()
            del x_input_map_multi

        elif config.PSM:
            siz = src.size()  # bs,T,F
            assert len(siz) == 3
            # topk_max = config.MAX_MIX  # 最多可能的topk个数
            topk_max = 2  # 最多可能的topk个数
            x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous()  # bs,topk,T,F
            feas_tgt_tmp = feas_tgt.view(siz[0], -1, siz[1], siz[2])

            IRM=feas_tgt_tmp/(x_input_map_multi+1e-15)

            angle_tgt=models.rank_feas(raw_tgt, train_data['multi_spk_angle_list']).view(siz[0],-1,siz[1],siz[2])
            angle_mix=Variable(torch.from_numpy(np.array(train_data['mix_angle']))).unsqueeze(1).expand(siz[0], topk_max, siz[1], siz[2]).contiguous()
            ang=np.cos(angle_mix-angle_tgt)
            ang=np.clip(ang,0,None)

            # feas_tgt = x_input_map_multi *np.clip(IRM.numpy()*ang,0,1) # bs,topk,T,F
            # feas_tgt = x_input_map_multi *IRM*ang # bs,topk,T,F
            feas_tgt = feas_tgt.view(siz[0],-1,siz[1],siz[2])*ang # bs,topk,T,F
            feas_tgt = feas_tgt.view(-1, siz[1], siz[2])  # bs*topk,T,F
            del x_input_map_multi

        if use_cuda:
            src = src.cuda().transpose(0, 1)
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()
            if config.WFM:
                WFM_mask = WFM_mask.cuda()

        # try:
        if 1 and len(opt.gpus) > 1:
            # samples,  predicted_masks = model.module.pit_sample(src, src_len, dict_spk2idx, tgt,
            #                                                                         beam_size=config.beam_size)
            predicted_masks, enc_attn_list = model.module(src, src_len, tgt, tgt_len,
                                              dict_spk2idx)  # 这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        else:
            # samples,  predicted_masks = model.pit_sample(src, src_len, dict_spk2idx, tgt,
            #                                                                  beam_size=config.beam_size)
            predicted_masks, enc_attn_list = model(src, src_len, tgt, tgt_len,
                                              dict_spk2idx)  # 这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        # samples=samples.max(2)[1].data.cpu().numpy()
        # except:
        #     continue

        # '''
        # expand the raw mixed-features to topk_max channel.
        src = src.transpose(0, 1)
        siz = src.size()
        assert len(siz) == 3
        # if samples[0][-1] != dict_spk2idx['<EOS>']:
        #     print '*'*40+'\nThe model is far from good. End the evaluation.\n'+'*'*40
        #     break
        topk_max = config.MAX_MIX
        x_input_map_multi = torch.unsqueeze(src, 1).expand(siz[0], topk_max, siz[1], siz[2])

        if 1 and len(opt.gpus) > 1:
            ss_loss,best_pmt = model.module.separation_pit_loss(x_input_map_multi, predicted_masks, feas_tgt, )
        else:
            ss_loss,best_pmt = model.separation_pit_loss(x_input_map_multi, predicted_masks, feas_tgt)
        print(('loss for ss,this batch:', ss_loss.cpu().item()))
        print('best perms for this batch:', best_pmt)
        lera.log({
            'ss_loss_' + test_or_valid: ss_loss.cpu().item(),
        })
        del ss_loss

        # '''''
        if 1 and batch_idx <= (500 / config.batch_size):  # only the former batches counts the SDR
            predicted_maps = predicted_masks.transpose(0,1) * x_input_map_multi
            predicted_maps = predicted_maps.view(-1,mix_speech_len,speech_fre)
            # predicted_maps=Variable(feas_tgt)
            utils.bss_eval2(config, predicted_maps, eval_data['multi_spk_fea_list'], raw_tgt, eval_data,
                            dst='batch_output_test')
            del predicted_maps, predicted_masks, x_input_map_multi
            try:
                sdr_aver_batch, sdri_aver_batch=  bss_test.cal('batch_output_test/')
                SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
                SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
            except(AssertionError):
                print('Errors in calculating the SDR')
            print(('SDR_aver_now:', SDR_SUM.mean()))
            print(('SRi_aver_now:', SDRi_SUM.mean()))
            lera.log({'SDR sample'+test_or_valid: SDR_SUM.mean()})
            lera.log({'SDRi sample'+test_or_valid: SDRi_SUM.mean()})
            writer.add_scalars('scalar/loss',{'SDR_sample_'+test_or_valid:sdr_aver_batch},updates)
            # raw_input('Press any key to continue......')
        elif batch_idx == (200 / config.batch_size) + 1 and SDR_SUM.mean() > best_SDR:  # only record the best SDR once.
            print(('Best SDR from {}---->{}'.format(best_SDR, SDR_SUM.mean())))
            best_SDR = SDR_SUM.mean()
            # save_model(log_path+'checkpoint_bestSDR{}.pt'.format(best_SDR))

        # '''
        # candidate += [convertToLabels(dict_idx2spk, s, dict_spk2idx['<EOS>']) for s in samples]
        # source += raw_src
        # reference += raw_tgt
        # print(('samples:', samples))
        # print(('can:{}, \nref:{}'.format(candidate[-1 * config.batch_size:], reference[-1 * config.batch_size:])))
        # alignments += [align for align in alignment]
        batch_idx += 1

        result = utils.eval_metrics(reference, candidate, dict_spk2idx, log_path)
        print(('hamming_loss: %.8f | micro_f1: %.4f |recall: %.4f | precision: %.4f'
                   % (result['hamming_loss'], result['micro_f1'], result['micro_recall'], result['micro_precision'], )))

    score = {}
    result = utils.eval_metrics(reference, candidate, dict_spk2idx, log_path)
    logging_csv([e, updates, result['hamming_loss'], \
                 result['micro_f1'], result['micro_precision'], result['micro_recall'],SDR_SUM.mean()])
    print(('hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1'])))
    score['hamming_loss'] = result['hamming_loss']
    score['micro_f1'] = result['micro_f1']
    1/0
    return score


# Convert `idx` to labels. If index `stop` is reached, convert it and return.
def convertToLabels(dict, idx, stop):
    labels = []

    for i in idx:
        i = int(i)
        if i == stop:
            break
        labels += [dict[i]]

    return labels


def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}

    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch + 1):
        if not opt.notrain:
            train(i)
        else:
            eval(i)
            # eval_recu(i)
    for metric in config.metric:
        print('updates:',updates)
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    # freeze_support()
    main()
