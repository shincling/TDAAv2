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
from predata_fromList_123 import prepare_data,prepare_datasize
import bss_test

import os
import argparse
import time
import json
import collections
import codecs


#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
# parser.add_argument('-restore', default='best_f1_v4.pt', type=str,
#                     help="restore checkpoint")
# parser.add_argument('-restore', default='best_f1_ct_v1.pt', type=str,
#                     help="restore checkpoint")
# parser.add_argument('-restore', default='best_f1_globalemb5.pt', type=str,
#                     help="restore checkpoint")
# parser.add_argument('-restore', default='best_schimit_v2.pt', type=str,
#                     help="restore checkpoint")
parser.add_argument('-restore', default='best_schimit_mix_v0.pt', type=str,
                    help="restore checkpoint")
# parser.add_argument('-restore', default='best_f1_WFM600.pt', type=str,
#                     help="restore checkpoint")
# parser.add_argument('-restore', default=None, type=str,
#                     help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', default=False, type=bool,
                    help="load pretrain embedding")
parser.add_argument('-notrain', default=False, type=bool,
                    help="train or not")
parser.add_argument('-limit', default=0, type=int,
                    help="data limit")
parser.add_argument('-log', default='plus', type=str,
                    help="log directory")
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

spk_global_gen=prepare_data(mode='global',train_or_test='train') #写一个假的数据生成，可以用来写模型先
global_para=spk_global_gen.next()
print global_para
spk_all_list,dict_spk2idx,dict_idx2spk,mix_speech_len,speech_fre,total_frames,spk_num_total,batch_total=global_para
config.speech_fre=speech_fre
config.mix_speech_len=mix_speech_len
del spk_global_gen
num_labels=len(spk_all_list)
print('loading the global setting cost: %.3f' % (time.time()-start_time))


if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None

# model
print('building model...\n')
# 这个用法有意思，实际是 调了model.seq2seq 并且运行了最后这个括号里的五个参数的方法。(初始化了一个对象也就是）
model = getattr(models, opt.model)(config, speech_fre, mix_speech_len, num_labels, use_cuda,
                       pretrain=pretrain_embed, score_fn=opt.score)

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())
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

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

total_loss, start_time = 0, time.time()
total_loss_sgm,total_loss_ss= 0 , 0
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))

with open(opt.label_dict_file, 'r') as f:
    label_dict = json.load(f)

# train
def train(epoch):
    e = epoch
    model.train()

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

    if opt.model == 'gated': 
        model.current_epoch = epoch

    global e, updates, total_loss, start_time, report_total, total_loss_sgm, total_loss_ss

    train_data_gen=prepare_data('once','train')
    # for raw_src, src, src_len, raw_tgt, tgt, tgt_len in trainloader:
    while True:
        train_data=train_data_gen.next()
        if train_data==False:
            break #如果这个epoch的生成器没有数据了，直接进入下一个epoch

        src = Variable(torch.from_numpy(train_data['mix_feas']))
        # raw_tgt = [spk.keys() for spk in train_data['multi_spk_fea_list']]
        raw_tgt = [sorted(spk.keys()) for spk in train_data['multi_spk_fea_list']]
        feas_tgt=models.rank_feas(raw_tgt,train_data['multi_spk_fea_list']) #这里是目标的图谱

        # 要保证底下这几个都是longTensor(长整数）
        tgt = Variable(torch.from_numpy(np.array([[0]+[dict_spk2idx[spk] for spk in spks]+[dict_spk2idx['<EOS>']] for spks in raw_tgt],dtype=np.int))).transpose(0,1) #转换成数字，然后前后加开始和结束符号。
        src_len = Variable(torch.LongTensor(config.batch_size).zero_()+mix_speech_len).unsqueeze(0)
        tgt_len = Variable(torch.LongTensor(config.batch_size).zero_()+len(train_data['multi_spk_fea_list'][0])).unsqueeze(0)
        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()

        model.zero_grad()
        # optim.optimizer.zero_grad()
        outputs, targets, multi_mask = model(src, src_len, tgt, tgt_len) #这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
        print 'mask size:',multi_mask.size()
        sgm_loss, num_total, num_correct = model.compute_loss(outputs, targets, opt.memory)
        print 'loss for SGM,this batch:',sgm_loss.data[0]/num_total

        # expand the raw mixed-features to topk channel.
        siz=src.size()
        assert len(siz)==3
        topk=feas_tgt.size()[1]
        x_input_map_multi=torch.unsqueeze(src,1).expand(siz[0],topk,siz[1],siz[2])
        ss_loss = model.separation_loss(x_input_map_multi, multi_mask, feas_tgt)

        loss=sgm_loss+ss_loss
        loss.backward()
        # print 'totallllllllllll loss:',loss
        total_loss_sgm += sgm_loss.data[0]
        total_loss_ss += ss_loss.data[0]
        total_loss += loss.data[0]
        report_total += num_total
        optim.step()
        updates += 1
        if updates%30==0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss this batch: %6.3f,sgm loss: %6.6f,ss loss: %6.6f\n"
                    % (time.time()-start_time, epoch, updates, loss / num_total, total_loss_sgm/30.0, total_loss_ss/30.0))
            total_loss_sgm, total_loss_ss = 0, 0

        # continue

        if 1 or updates % config.eval_interval == 0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f\n"
                    % (time.time()-start_time, epoch, updates, total_loss / report_total))
            print('evaluating after %d updates...\r' % updates)
            score = eval(epoch)
            for metric in config.metric:
                scores[metric].append(score[metric])
                if metric == 'micro_f1' and score[metric] >= max(scores[metric]):  
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
                if metric == 'hamming_loss' and score[metric] <= min(scores[metric]):
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')

            model.train()
            total_loss = 0
            start_time = 0
            report_total = 0

        if updates % config.save_interval == 0:  
            save_model(log_path+'checkpoint_v2.pt')


def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    eval_data_gen=prepare_data('once','test',2,2)
    # for raw_src, src, src_len, raw_tgt, tgt, tgt_len in validloader:
    SDR_SUM=np.array([])
    batch_idx=0
    while True:
    # for ___ in range(100):
        print '-'*30
        eval_data=eval_data_gen.next()
        if eval_data==False:
            break #如果这个epoch的生成器没有数据了，直接进入下一个epoch
        src = Variable(torch.from_numpy(eval_data['mix_feas']))

        raw_tgt = [sorted(spk.keys()) for spk in eval_data['multi_spk_fea_list']]
        top_k=len(raw_tgt[0])
        # 要保证底下这几个都是longTensor(长整数）
        # tgt = Variable(torch.from_numpy(np.array([[0]+[dict_spk2idx[spk] for spk in spks]+[dict_spk2idx['<EOS>']] for spks in raw_tgt],dtype=np.int))).transpose(0,1) #转换成数字，然后前后加开始和结束符号。
        tgt = Variable(torch.ones(top_k+2,config.batch_size)) # 这里随便给一个tgt，为了测试阶段tgt的名字无所谓其实。

        src_len = Variable(torch.LongTensor(config.batch_size).zero_()+mix_speech_len).unsqueeze(0)
        tgt_len = Variable(torch.LongTensor(config.batch_size).zero_()+len(eval_data['multi_spk_fea_list'][0])).unsqueeze(0)
        feas_tgt=models.rank_feas(raw_tgt,eval_data['multi_spk_fea_list']) #这里是目标的图谱
        if config.WFM:
            tmp_size=feas_tgt.size()
            assert len(tmp_size)==4
            feas_tgt_sum=torch.sum(feas_tgt,dim=1,keepdim=True)
            feas_tgt_sum_square=(feas_tgt_sum*feas_tgt_sum).expand(tmp_size)
            feas_tgt_square=feas_tgt*feas_tgt
            WFM_mask=feas_tgt_square/feas_tgt_sum_square

        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()
            if config.WFM:
                WFM_mask= WFM_mask.cuda()
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            try:
                samples, alignment, hiddens, predicted_masks = model.beam_sample(src, src_len, dict_spk2idx, tgt, beam_size=config.beam_size)
            except Exception,info:
                print '**************Error occurs here************:', info
                continue

        if config.top1:
            predicted_masks=torch.cat([predicted_masks,1-predicted_masks],1)

        # '''
        # expand the raw mixed-features to topk channel.
        siz=src.size()
        assert len(siz)==3
        topk=feas_tgt.size()[1]
        x_input_map_multi=torch.unsqueeze(src,1).expand(siz[0],topk,siz[1],siz[2])
        if config.WFM:
            feas_tgt=x_input_map_multi.data*WFM_mask
        ss_loss = model.separation_loss(x_input_map_multi, predicted_masks, feas_tgt)
        print 'loss for ss,this batch:',ss_loss.data[0]
        del ss_loss,hiddens

        if batch_idx<=(500/config.batch_size): #only the former batches counts the SDR
            predicted_maps=predicted_masks*x_input_map_multi
            utils.bss_eval(config, predicted_maps,eval_data['multi_spk_fea_list'], raw_tgt, eval_data, dst='batch_outputtt_nomask_hiddenall')
            del predicted_maps,predicted_masks,x_input_map_multi
            SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_outputtt_nomask_hiddenall/'))
            print 'SDR_aver_now:',SDR_SUM.mean()
            # raw_input('Press any key to continue......')

        # '''
        candidate += [convertToLabels(dict_idx2spk,s, dict_spk2idx['<EOS>']) for s in samples]
        # source += raw_src
        reference += raw_tgt
        print 'samples:',samples
        print 'can:{}, \nref:{}'.format(candidate[-1*config.batch_size:],reference[-1*config.batch_size:])
        alignments += [align for align in alignment]
        batch_idx+=1

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}
    result = utils.eval_metrics(reference, candidate,dict_spk2idx, log_path)
    logging_csv([e, updates, result['hamming_loss'], \
                result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1']))
    score['hamming_loss'] = result['hamming_loss']
    score['micro_f1'] = result['micro_f1']
    return score

# Convert `idx` to labels. If index `stop` is reached, convert it and return.
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
    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch+1):
        if not opt.notrain:
            train(i)
        else:
            eval(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
