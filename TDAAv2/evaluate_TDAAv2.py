#coding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
import lr_scheduler as L
from predata_given import prepare_data,prepare_datasize
import bss_test

import os
import argparse
import time
import json
import collections
import lera
import soundfile as sf
import librosa
import code

#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[2], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='best_f1_globalemb12.pt', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', default=False, type=bool,
                    help="load pretrain embedding")
parser.add_argument('-notrain', default=1, type=bool,
                    help="train or not")
parser.add_argument('-limit', default=0, type=int,
                    help="data limit")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
parser.add_argument('-input_dir', default='./given_wav', type=str,
                    help="The path to the mixed Audio file.")
parser.add_argument('-output_dir', default='./given_output', type=str,
                    help="The path to save the separated Audio file.")
parser.add_argument('-unk', default=False, type=bool,
                    help="replace unk")
parser.add_argument('-memory', default=False, type=bool,
                    help="memory efficiency")
parser.add_argument('-label_dict_file', default='./data/data/rcv1.json', type=str,
                    help="label_dict")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore: 
    print('loading checkpoint...\n',opt.restore)
    checkpoints = torch.load(opt.restore)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)

# data
print('loading data...\n')
start_time = time.time()

# spk_global_gen=prepare_data(mode='global',train_or_test='train') #写一个假的数据生成，可以用来写模型先
# global_para=spk_global_gen.next()
# print global_para
# spk_all_list,dict_spk2idx,dict_idx2spk,mix_speech_len,speech_fre,total_frames,spk_num_total,batch_total=global_para
# del spk_global_gen
config.speech_fre=129
config.mix_speech_len=626
num_labels=103
dict_spk2idx={'<BOS>':0,'<EOS>':num_labels-1}
print('loading the global setting cost: %.3f' % (time.time()-start_time))


if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None

# model
print('building model...\n')
# 这个用法有意思，实际是 调了model.seq2seq 并且运行了最后这个括号里的五个参数的方法。(初始化了一个对象也就是）
model = getattr(models, opt.model)(config, config.speech_fre, config.mix_speech_len, num_labels, use_cuda,
                       pretrain=pretrain_embed, score_fn=opt.score)
if config.is_dis:
    model_dis = models.separation_dis.Discriminator().cuda()
    func_dis = torch.nn.MSELoss()

if opt.restore:
    model.load_state_dict(checkpoints['model'])
    if 0 and config.is_dis:
        model_dis.load_state_dict(checkpoints['model_dis'])
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

optim.set_parameters(model.parameters())
if config.is_dis:
    if 0 and opt.restore:
        optim_dis = checkpoints['optim_dis']
    else:
        optim_dis = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                      lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
        optim_dis.set_parameters(model_dis.parameters())
    scheduler_dis = L.CosineAnnealingLR(optim_dis.optimizer, T_max=config.epoch)

if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt') # 这种方式也值得学习，单独写一个logging的函数，直接调用，既print，又记录到Log文件里。
logging_csv = utils.logging_csv(log_path+'record.csv') 
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")
if config.is_dis:
    logging(repr(model_dis)+"\n\n")

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

if 1 and opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

if config.MLMSE:
    if opt.restore and 'Var' in checkpoints:
        Var = checkpoints['Var']
    else:
        Var = None

total_loss, start_time = 0, time.time()
total_loss_sgm,total_loss_ss= 0 , 0
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))
best_SDR=0.0

with open(opt.label_dict_file, 'r') as f:
    label_dict = json.load(f)

# train
lera.log_hyperparams({
  'title': unicode('MLMSE train with hidden Relu+Order') ,
  'updates':updates,
  'batch_size': config.batch_size,
  'WFM':config.WFM,
  'MLMSE':config.MLMSE,
  'top1':config.top1 ,#控制是否用top-1的方法来生产第二个通道
  'global_emb':config.global_emb,
  # 'spk_emb_size':config.SPK_EMB_SIZE, #注意这个东西需要和上面的spk_emb（如果是global_emb的话）,否认则就跟decoder的hidden_size对应
  # 'hidden_mix':config.hidden_mix,#这个模式是不采用global emb的时候，把hidden和下一步的embeding一起cat起来作为ss的输入进去。
  'schmidt': config.schmidt,
  'log path': unicode(log_path),
  'selfTune':config.is_SelfTune,
  'is_dis':config.is_dis,
  'cnn':config.speech_cnn_net,#是否采用CNN的结构来抽取
  'relitu':config.relitu,
  'ct_recu':config.ct_recu,#控制是否采用att递减的ct构成
})

def bss_eval(config, predict_multi_map,y_map_gtruth,train_data, name, dst='batch_output'):
    # dst='batch_output'
    name.replace('.wav','')
    if not os.path.exists(dst):
        os.makedirs(dst)
    y_map_gtruth=[y_map_gtruth]

    # 对于每个sample
    sample_idx=0 #代表一个batch里的依次第几个
    for each_pre,each_trueVector in zip(predict_multi_map,y_map_gtruth):
        _mix_spec=train_data['mix_phase'][sample_idx]
        # _mix_spec=_mix_spec[:config.buffer_shift]
        phase_mix = np.angle(_mix_spec)
        for idx,one_cha in enumerate(each_trueVector):
            this_spk=one_cha
            y_pre_map=each_pre[idx].data.cpu().numpy()
            _pred_spec = y_pre_map * np.exp(1j * phase_mix)
            wav_pre=librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            min_len =  len(wav_pre)
            sf.write(dst+'/{}_{}_pre.wav'.format(name,this_spk),wav_pre[:min_len],config.FRAME_RATE,)
        sf.write(dst+'/{}_True_mix.wav'.format(name),train_data['mix_wav'][sample_idx][:min_len],config.FRAME_RATE,)
        sample_idx+=1


def eval(epoch):
    model.eval()
    opt.input_dir= opt.input_dir if opt.input_dir[-1]=='/' else opt.input_dir+'/'
    opt.output_dir= opt.output_dir if opt.output_dir[-1]=='/' else opt.output_dir+'/'
    print 'Begin to watch the dir:',opt.input_dir
    print '-'*30
    reference, candidate, source, alignments = [], [], [], []
    # for raw_src, src, src_len, raw_tgt, tgt, tgt_len in validloader:
    SDR_SUM=np.array([])
    batch_idx=0
    mix_files=[]
    global best_SDR,Var
    while True:
        time.sleep(5)
        now_files=os.listdir(opt.input_dir)
        if len(now_files)==0:
            continue
        else:
            for av in now_files:
                if av.endswith('wav') and av not in mix_files:
                    print 'Begin to process the aduio:',av
                    mix_files.append(av)

                    eval_data=prepare_data('once','test',config.MIN_MIX,config.MAX_MIX,opt.input_dir+av)

                    src = Variable(torch.from_numpy(eval_data['mix_feas']))

                    raw_tgt = ['spk1','spk2']
                    top_k=len(raw_tgt[0])
                    # 要保证底下这几个都是longTensor(长整数）
                    tgt = Variable(torch.ones(top_k+2,config.batch_size)) # 这里随便给一个tgt，为了测试阶段tgt的名字无所谓其实。

                    src_len = Variable(torch.LongTensor(config.batch_size).zero_()+config.mix_speech_len).unsqueeze(0)

                    if use_cuda:
                        src = src.cuda().transpose(0,1)
                        tgt = tgt.cuda()
                        src_len = src_len.cuda()

                    if 1 and len(opt.gpus) > 1:
                        # samples, alignment = model.module.sample(src, src_len)
                        samples, alignment, hiddens, predicted_masks = model.module.beam_sample(src, src_len, dict_spk2idx, tgt, beam_size=config.beam_size)
                    else:
                        samples, alignment, hiddens, predicted_masks = model.beam_sample(src, src_len, dict_spk2idx, tgt, beam_size=config.beam_size)

                    if config.top1:
                        predicted_masks=torch.cat([predicted_masks,1-predicted_masks],1)

                    # '''
                    # expand the raw mixed-features to topk channel.
                    src = src.transpose(0,1)
                    siz=src.size()
                    assert len(siz)==3
                    topk=2
                    x_input_map_multi=torch.unsqueeze(src,1).expand(siz[0],topk,siz[1],siz[2])

                    # '''''
                    predicted_maps=predicted_masks*x_input_map_multi
                    bss_eval(config, predicted_maps, raw_tgt, eval_data, av, dst=opt.output_dir)

def convertToLabels(dict, idx, stop):
    labels = []

    for i in idx:
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

    if config.MLMSE:
        checkpoints['Var']=Var
    if config.is_dis:
        model_dis_state_dict = model_dis.module.state_dict() if len(opt.gpus) > 1 else model_dis.state_dict()
        checkpoints['model_dis']=model_dis_state_dict
        checkpoints['optim_dis']=optim_dis
    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch+1):
        if not opt.notrain:
            pass
        else:
            eval(i)

if __name__ == '__main__':
    main()
