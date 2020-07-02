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
parser.add_argument('-gpus', default=[2,0,3], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
# parser.add_argument('-restore', default='TDAAv3_PIT_30001.pt', type=str,
# parser.add_argument('-restore', default='data/data/log/2020-02-24-08:02:28/Transformer_PIT_2ch_6001.pt', type=str,
parser.add_argument('-restore', default='data/data/log/2020-02-26-13:45:07/Transformer_PIT_2ch_22001.pt', type=str,
# parser.add_argument('-restore', default=None, type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-notrain', default=1, type=bool,
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

def average_model_pytorch(ifiles, ofile):
    import collections
    import torch
    omodel = collections.OrderedDict()
    device = torch.device('cpu')

    for ifile in ifiles:
        # checkpoint = torch.load(ifile, map_location=device)
        checkpoint = torch.load(ifile,map_location={'cuda:4':'cuda:0'})
        state_dict =  checkpoint['model']
        for i, (key, value) in enumerate(state_dict.items()):
            if key not in omodel:
                omodel[key] = value
            else:
                omodel[key] += value
    for key in omodel.keys():
        omodel[key] = omodel[key] / len(ifiles)

    return omodel


# checkpoint
if opt.restore:
    print(('loading checkpoint...\n', opt.restore))
    checkpoints = torch.load(opt.restore,map_location={'cuda:4':'cuda:0'})
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
# model = getattr(models, opt.model)(config, speech_fre, mix_speech_len, num_labels, use_cuda, None, opt.score_fc)

# two channel
model = getattr(models, opt.model)(config, speech_fre, mix_speech_len, num_labels, use_cuda, None, opt.score_fc)

if config.use_center_loss:
    center_loss=models.CenterLoss(num_classes=num_labels,feat_dim=config.SPK_EMB_SIZE,use_gpu=True)
    print(('Here we use center loss:',center_loss))

if opt.restore:
    if 0: #given some checkpoints to do model averaging
        checkpoints_list=['/data1/shijing_data/2020-02-14-04:58:17//Transformer_PIT_9001.pt',
              '/data1/shijing_data/2020-02-14-04:58:17//Transformer_PIT_10001.pt',
              '/data1/shijing_data/2020-02-14-04:58:17//Transformer_PIT_11001.pt',
              # '/data1/shijing_data/2020-02-14-04:58:17//Transformer_PIT_15001.pt',
              # '/data1/shijing_data/2020-02-14-04:58:17//Transformer_PIT_14001.pt',
              ]
        checkpoints_list=['/data1/shijing_data/2020-02-14-05:43:10/Transformer_PIT_60001.pt',
                          '/data1/shijing_data/2020-02-14-05:43:10/Transformer_PIT_59001.pt',
                          '/data1/shijing_data/2020-02-14-05:43:10/Transformer_PIT_58001.pt',
                          '/data1/shijing_data/2020-02-14-05:43:10/Transformer_PIT_57001.pt',
                          ]
        print('Average over list:',checkpoints_list)
        averge_model=average_model_pytorch(checkpoints_list,None)
        model.load_state_dict(averge_model)
    else:
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
    # scheduler = L.StepLR(optim.optimizer, step_size=15, gamma=0.3)
    scheduler = L.StepLR(optim.optimizer, step_size=10, gamma=0.9)

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
    'title': str('Transformer PIT 2CH time loss '),
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
    '2channel':config.is_two_channel,
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


    train_data_gen = prepare_data('once', 'train')
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

        src = Variable(torch.from_numpy(train_data['mix_complex_two_channel'])) # bs,T,F,2 both real and imag values
        raw_tgt=train_data['batch_order']
        feas_tgt = models.rank_feas(raw_tgt, train_data['multi_spk_wav_list'])  # 这里是目标的图谱,bs*Topk,time_len

        padded_mixture, mixture_lengths, padded_source = train_data['tas_zip']
        padded_mixture=torch.from_numpy(padded_mixture).float()
        mixture_lengths=torch.from_numpy(mixture_lengths)
        padded_source=torch.from_numpy(padded_source).float()

        padded_mixture = padded_mixture.cuda().transpose(0,1)
        mixture_lengths = mixture_lengths.cuda()
        padded_source = padded_source.cuda()

        # 要保证底下这几个都是longTensor(长整数）
        tgt_max_len = config.MAX_MIX + 2  # with bos and eos.
        tgt = Variable(torch.from_numpy(np.array(
            [[0] + [dict_spk2idx[spk] for spk in spks] + (tgt_max_len - len(spks) - 1) * [dict_spk2idx['<EOS>']] for
             spks in raw_tgt], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。
        # tgt = Variable(torch.from_numpy(np.array([[0,1,2,102] for __ in range(config.batch_size)], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。
        src_len = Variable(torch.LongTensor(config.batch_size).zero_() + mix_speech_len).unsqueeze(0)
        tgt_len = Variable(
            torch.LongTensor([len(one_spk) for one_spk in train_data['multi_spk_fea_list']])).unsqueeze(0)

        if use_cuda:
            src = src.cuda().transpose(0, 1)
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()

        model.zero_grad()
        if config.use_center_loss:
            center_loss.zero_grad()

        multi_mask_real,multi_mask_imag, enc_attn_list = model(src, src_len, tgt, tgt_len,
                                             dict_spk2idx)  # 这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        multi_mask_real=multi_mask_real.transpose(0,1)
        multi_mask_imag=multi_mask_imag.transpose(0,1)
        src_real=src[:,:,:,0].transpose(0,1) # bs,T,F
        src_imag=src[:,:,:,1].transpose(0,1) # bs,T,F
        print('mask size for real/imag:', multi_mask_real.size()) # bs,topk,T,F, 已经压缩过了
        print('mixture size for real/imag:', src_real.size()) # bs,T,F

        predicted_maps0_real=multi_mask_real[:,0]*src_real - multi_mask_imag[:,0]*src_imag #bs,T,F
        predicted_maps0_imag=multi_mask_real[:,0]*src_imag + multi_mask_imag[:,0]*src_real #bs,T,F
        predicted_maps1_real=multi_mask_real[:,1]*src_real - multi_mask_imag[:,1]*src_imag #bs,T,F
        predicted_maps1_imag=multi_mask_real[:,1]*src_imag + multi_mask_imag[:,1]*src_real #bs,T,F

        stft_matrix_spk0=torch.cat((predicted_maps0_real.unsqueeze(-1),predicted_maps0_imag.unsqueeze(-1)),3).transpose(1,2) # bs,F,T,2
        stft_matrix_spk1=torch.cat((predicted_maps1_real.unsqueeze(-1),predicted_maps1_imag.unsqueeze(-1)),3).transpose(1,2) # bs,F,T,2
        wav_spk0 = models.istft_irfft(stft_matrix_spk0, length=config.MAX_LEN, hop_length=config.FRAME_SHIFT, win_length=config.FRAME_LENGTH, window='hann' )
        wav_spk1 = models.istft_irfft(stft_matrix_spk1, length=config.MAX_LEN, hop_length=config.FRAME_SHIFT, win_length=config.FRAME_LENGTH, window='hann' )
        predict_wav = torch.cat((wav_spk0.unsqueeze(1),wav_spk1.unsqueeze(1)),1) # bs,topk,time_len
        if 1 and len(opt.gpus) > 1:
            ss_loss, pmt_list, max_snr_idx, *__ = model.module.separation_tas_loss(padded_mixture,predict_wav, padded_source, mixture_lengths)
        else:
            ss_loss, pmt_list, max_snr_idx, *__ = model.separation_tas_loss(padded_mixture,predict_wav, padded_source, mixture_lengths)

        best_pmt = [list(pmt_list[int(mm)].data.cpu().numpy()) for mm in max_snr_idx]
        print('loss for SS,this batch:', ss_loss.cpu().item())
        print('best perms for this batch:', best_pmt)
        writer.add_scalars('scalar/loss',{'ss_loss':ss_loss.cpu().item()},updates)

        loss = ss_loss
        loss.backward()

        total_loss_ss += ss_loss.cpu().item()
        lera.log({
            'ss_loss': ss_loss.cpu().item(),
        })

        if epoch>20 and updates>5 and updates % config.eval_interval in [0, 1, 2, 3, 4]:
            utils.bss_eval_tas(config,predict_wav, train_data['multi_spk_fea_list'], raw_tgt, train_data, dst=log_path+'batch_output')
            sdr_aver_batch, snri_aver_batch=  bss_test.cal(log_path+'batch_output/')
            lera.log({'SDR sample': sdr_aver_batch})
            lera.log({'SI-SNRi sample': snri_aver_batch})
            writer.add_scalars('scalar/loss',{'SDR_sample':sdr_aver_batch,'SDRi_sample': snri_aver_batch},updates)
            SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
            SDRi_SUM = np.append(SDRi_SUM,snri_aver_batch)
            print(('SDR_aver_now:', SDR_SUM.mean()))
            print(('SNRi_aver_now:', SDRi_SUM.mean()))

            # Heatmap here
            # n_layer个 (head*bs) x lq x dk
            '''
            import matplotlib.pyplot as plt
            ax = plt.gca()
            ax.invert_yaxis()

            raw_src=models.rank_feas(raw_tgt, train_data['multi_spk_fea_list'])
            att_idx=1
            att = enc_attn_list[-1].view(config.trans_n_head,config.batch_size,mix_speech_len,mix_speech_len).data.cpu().numpy()[:,att_idx]
            for head in range(config.trans_n_head):
                xx=att[head]
                plt.matshow(xx, cmap=plt.cm.hot, vmin=0,vmax=0.05)
                plt.colorbar()
                plt.savefig(log_path+'batch_output/'+'head_{}.png'.format(head))
            plt.matshow(raw_src[att_idx*2+0].transpose(0,1), cmap=plt.cm.hot, vmin=0,vmax=2)
            plt.colorbar()
            plt.savefig(log_path+'batch_output/'+'source0.png')
            plt.matshow(raw_src[att_idx*2+1].transpose(0,1), cmap=plt.cm.hot, vmin=0,vmax=2)
            plt.colorbar()
            plt.savefig(log_path+'batch_output/'+'source1.png')
            1/0
            '''

        total_loss += loss.cpu().item()
        optim.step()

        updates += 1
        if updates % 30 == 0:
            logging(
                "time: %6.3f, epoch: %3d, updates: %8d, train loss this batch: %6.3f,ss loss: %6.6f\n"
                % (time.time() - start_time, epoch, updates, loss , total_loss_ss / 30.0))
            total_loss_sgm, total_loss_ss = 0, 0

        # continue

        if 1 and updates % config.save_interval == 1:
            save_model(log_path + 'Transformer_PIT_2ch_{}.pt'.format(updates))

        if 0 and updates>0 and updates % config.eval_interval == 3 : #建议至少跑几个epoch再进行测试，否则模型还没学到东西，会有很多问题。
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f\n"
                    % (time.time() - start_time, epoch, updates, total_loss/config.eval_interval))
            print(('evaluating after %d updates...\r' % updates))
            eval(epoch,'valid') # eval的时候batch_size会变成1
            eval(epoch,'test') # eval的时候batch_size会变成1

            model.train()
            total_loss = 0
            start_time = 0
            report_total = 0
            report_correct = 0



def eval(epoch,test_or_valid='valid'):
    # config.batch_size=1
    global updates,model
    model.eval()
    # print '\n\n测试的时候请设置config里的batch_size为1！！！please set the batch_size as 1'
    reference, candidate, source, alignments = [], [], [], []
    e = epoch
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
        src = Variable(torch.from_numpy(eval_data['mix_complex_two_channel'])) # bs,T,F,2 both real and imag values
        raw_tgt=eval_data['batch_order']
        feas_tgt = models.rank_feas(raw_tgt, eval_data['multi_spk_wav_list'])  # 这里是目标的图谱,bs*Topk,time_len

        padded_mixture, mixture_lengths, padded_source = eval_data['tas_zip']
        padded_mixture=torch.from_numpy(padded_mixture).float()
        mixture_lengths=torch.from_numpy(mixture_lengths)
        padded_source=torch.from_numpy(padded_source).float()

        padded_mixture = padded_mixture.cuda().transpose(0,1)
        mixture_lengths = mixture_lengths.cuda()
        padded_source = padded_source.cuda()

        # 要保证底下这几个都是longTensor(长整数）
        tgt = Variable(torch.from_numpy(np.array([[0,1,2,102] for __ in range(config.batch_size)], dtype=np.int))).transpose(0, 1)  # 转换成数字，然后前后加开始和结束符号。

        src_len = Variable(torch.LongTensor(config.batch_size).zero_() + mix_speech_len).unsqueeze(0)
        tgt_len = Variable(
            torch.LongTensor([len(one_spk) for one_spk in eval_data['multi_spk_fea_list']])).unsqueeze(0)

        if use_cuda:
            src = src.cuda().transpose(0, 1)
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()

        model.zero_grad()
        if config.use_center_loss:
            center_loss.zero_grad()

        multi_mask_real,multi_mask_imag, enc_attn_list = model(src, src_len, tgt, tgt_len,
                                                               dict_spk2idx)  # 这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        multi_mask_real=multi_mask_real.transpose(0,1)
        multi_mask_imag=multi_mask_imag.transpose(0,1)
        src_real=src[:,:,:,0].transpose(0,1) # bs,T,F
        src_imag=src[:,:,:,1].transpose(0,1) # bs,T,F
        print('mask size for real/imag:', multi_mask_real.size()) # bs,topk,T,F, 已经压缩过了
        print('mixture size for real/imag:', src_real.size()) # bs,T,F

        predicted_maps0_real=multi_mask_real[:,0]*src_real - multi_mask_imag[:,0]*src_imag #bs,T,F
        predicted_maps0_imag=multi_mask_real[:,0]*src_imag + multi_mask_imag[:,0]*src_real #bs,T,F
        predicted_maps1_real=multi_mask_real[:,1]*src_real - multi_mask_imag[:,1]*src_imag #bs,T,F
        predicted_maps1_imag=multi_mask_real[:,1]*src_imag + multi_mask_imag[:,1]*src_real #bs,T,F

        stft_matrix_spk0=torch.cat((predicted_maps0_real.unsqueeze(-1),predicted_maps0_imag.unsqueeze(-1)),3).transpose(1,2) # bs,F,T,2
        stft_matrix_spk1=torch.cat((predicted_maps1_real.unsqueeze(-1),predicted_maps1_imag.unsqueeze(-1)),3).transpose(1,2) # bs,F,T,2
        wav_spk0 = models.istft_irfft(stft_matrix_spk0, length=config.MAX_LEN, hop_length=config.FRAME_SHIFT, win_length=config.FRAME_LENGTH, window='hann' )
        wav_spk1 = models.istft_irfft(stft_matrix_spk1, length=config.MAX_LEN, hop_length=config.FRAME_SHIFT, win_length=config.FRAME_LENGTH, window='hann' )
        predict_wav = torch.cat((wav_spk0.unsqueeze(1),wav_spk1.unsqueeze(1)),1) # bs,topk,time_len
        if 1 and len(opt.gpus) > 1:
            ss_loss, pmt_list, max_snr_idx, *__ = model.module.separation_tas_loss(padded_mixture,predict_wav, padded_source, mixture_lengths)
        else:
            ss_loss, pmt_list, max_snr_idx, *__ = model.separation_tas_loss(padded_mixture,predict_wav, padded_source, mixture_lengths)

        best_pmt = [list(pmt_list[int(mm)].data.cpu().numpy()) for mm in max_snr_idx]
        print('loss for SS,this batch:', ss_loss.cpu().item())
        print('best perms for this batch:', best_pmt)
        writer.add_scalars('scalar/loss',{'ss_loss':ss_loss.cpu().item()},updates)
        lera.log({
            'ss_loss_' + test_or_valid: ss_loss.cpu().item(),
        })
        writer.add_scalars('scalar/loss',{'ss_loss_'+test_or_valid:ss_loss.cpu().item()},updates+batch_idx)
        del ss_loss
        # if batch_idx>10:
        #     break

        if False: #this part is to test the checkpoints sequencially.
            batch_idx += 1
            if batch_idx%100==0:
                updates=updates+1000
                opt.restore='/data1/shijing_data/2020-02-14-04:58:17/Transformer_PIT_{}.pt'.format(updates)
                print('loading checkpoint...\n', opt.restore)
                checkpoints = torch.load(opt.restore)
                model.module.load_state_dict(checkpoints['model'])
                break
            continue
        # '''''
        if 1 and batch_idx<=(500/config.batch_size):
            utils.bss_eval_tas(config,predict_wav, eval_data['multi_spk_fea_list'], raw_tgt, eval_data, dst=log_path+'batch_output')
            sdr_aver_batch, snri_aver_batch=  bss_test.cal(log_path+'batch_output/')
            lera.log({'SDR sample': sdr_aver_batch})
            lera.log({'SI-SNRi sample': snri_aver_batch})
            writer.add_scalars('scalar/loss',{'SDR_sample':sdr_aver_batch,'SDRi_sample': snri_aver_batch},updates)
            SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
            SDRi_SUM = np.append(SDRi_SUM,snri_aver_batch)
            print(('SDR_aver_now:', SDR_SUM.mean()))
            print(('SNRi_aver_now:', SDRi_SUM.mean()))

        batch_idx += 1
        if batch_idx>100:
            break
        result = utils.eval_metrics(reference, candidate, dict_spk2idx, log_path)
        print(('hamming_loss: %.8f | micro_f1: %.4f |recall: %.4f | precision: %.4f'
                   % (result['hamming_loss'], result['micro_f1'], result['micro_recall'], result['micro_precision'], )))



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
