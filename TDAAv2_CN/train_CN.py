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
from predata_CN import prepare_data
import bss_test

import os
import argparse
import time
import json
import collections
import lera
import code

#config
parser = argparse.ArgumentParser(description='train_CN.py')

parser.add_argument('-config', default='config_CN.yaml', type=str,
                    help="config file")
# parser.add_argument('-gpus', default=range(8), nargs='+', type=int,
parser.add_argument('-gpus', default=[2], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='sscn_v01a_wfm.pt', type=str,
# parser.add_argument('-restore', default='sscn_v01a_264001.pt', type=str,
# parser.add_argument('-restore', default=None, type=str,
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
spk_all_list=global_para['all_spk']
dict_spk2idx=global_para['dict_spk_to_idx']
dict_idx2spk=global_para['dict_idx_to_spk']
speech_fre=global_para['num_fre']  # 语音频率
total_frames=global_para['num_frames']  # 语音长度
spk_num_total=global_para['total_spk_num']
batch_total=global_para['total_batch_num']

config.speech_fre=speech_fre
mix_speech_len=total_frames
config.mix_speech_len=total_frames
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
  'title': unicode('SS CN v0.1b') ,
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
def train(epoch):
    e = epoch
    model.train()
    SDR_SUM=np.array([])

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        if config.is_dis:
            scheduler_dis.step()
        lera.log({
            'lr': scheduler.get_lr()[0],
        })

    if opt.model == 'gated': 
        model.current_epoch = epoch

    global e, updates, total_loss, start_time, report_total, total_loss_sgm, total_loss_ss
    if config.MLMSE:
        global Var

    train_data_gen=prepare_data('once','train')
    # for raw_src, src, src_len, raw_tgt, tgt, tgt_len in trainloader:
    while True:
        try:
            train_data=train_data_gen.next()
            if train_data==False:
                print 'SDR_aver_epoch:',SDR_SUM.mean()
                break #如果这个epoch的生成器没有数据了，直接进入下一个epoch

            src = Variable(torch.from_numpy(train_data['mix_feas']))
            # raw_tgt = [spk.keys() for spk in train_data['multi_spk_fea_list']]
            raw_tgt = [sorted(spk.keys()) for spk in train_data['multi_spk_fea_list']]
            feas_tgt=models.rank_feas(raw_tgt,train_data['multi_spk_fea_list']) #这里是目标的图谱,aim_size,len,fre


            # 要保证底下这几个都是longTensor(长整数）
            tgt_max_len=config.MAX_MIX+2 # with bos and eos.
            tgt = Variable(torch.from_numpy(np.array([[0]+[dict_spk2idx[spk] for spk in spks]+(tgt_max_len-len(spks)-1)*[dict_spk2idx['<EOS>']] for spks in raw_tgt],dtype=np.int))).transpose(0,1) #转换成数字，然后前后加开始和结束符号。
            src_len = Variable(torch.LongTensor(config.batch_size).zero_()+mix_speech_len).unsqueeze(0)
            tgt_len = Variable(torch.LongTensor([len(one_spk) for one_spk in train_data['multi_spk_fea_list']])).unsqueeze(0)
            if use_cuda:
                src = src.cuda().transpose(0,1)
                tgt = tgt.cuda()
                src_len = src_len.cuda()
                tgt_len = tgt_len.cuda()
                feas_tgt = feas_tgt.cuda()

            model.zero_grad()
            # optim.optimizer.zero_grad()

            # aim_list 就是找到有正经说话人的地方的标号
            aim_list=(tgt[1:-1].transpose(0,1).contiguous().view(-1)!=dict_spk2idx['<EOS>']).nonzero().squeeze()
            aim_list=aim_list.data.cpu().numpy()

            outputs, targets, multi_mask = model(src, src_len, tgt, tgt_len,dict_spk2idx) #这里的outputs就是hidden_outputs，还没有进行最后分类的隐层，可以直接用
            print 'mask size:',multi_mask.size()

            if 1 and len(opt.gpus) > 1:
                sgm_loss, num_total, num_correct = model.module.compute_loss(outputs, targets, opt.memory)
            else:
                sgm_loss, num_total, num_correct = model.compute_loss(outputs, targets, opt.memory)
            print 'loss for SGM,this batch:',sgm_loss.data[0]/num_total

            src=src.transpose(0,1)
            # expand the raw mixed-features to topk_max channel.
            siz=src.size()
            assert len(siz)==3
            topk_max=config.MAX_MIX #最多可能的topk个数
            x_input_map_multi=torch.unsqueeze(src,1).expand(siz[0],topk_max,siz[1],siz[2]).contiguous().view(-1,siz[1],siz[2])
            x_input_map_multi=x_input_map_multi[aim_list]
            multi_mask=multi_mask.transpose(0,1)

            if 1 and len(opt.gpus) > 1:
                if config.MLMSE:
                    Var = model.module.update_var(x_input_map_multi, multi_mask, feas_tgt)
                    lera.log_image(u'Var weight',Var.data.cpu().numpy().reshape(config.speech_fre,config.speech_fre,1).repeat(3,2),clip=(-1,1))
                    ss_loss = model.module.separation_loss(x_input_map_multi, multi_mask, feas_tgt, Var)
                else:
                    ss_loss = model.module.separation_loss(x_input_map_multi, multi_mask, feas_tgt)
            else:
                ss_loss = model.separation_loss(x_input_map_multi, multi_mask, feas_tgt)

            loss=sgm_loss+5*ss_loss
            # dis_loss model
            if config.is_dis:
                dis_loss=models.loss.dis_loss(config,topk_max,model_dis,x_input_map_multi,multi_mask,feas_tgt,func_dis)
                loss = loss+dis_loss
                # print 'dis_para',model_dis.parameters().next()[0]
                # print 'ss_para',model.parameters().next()[0]

            loss.backward()
            # print 'totallllllllllll loss:',loss
            total_loss_sgm += sgm_loss.data[0]
            total_loss_ss += ss_loss.data[0]
            lera.log({
                'sgm_loss': sgm_loss.data[0],
                'ss_loss': ss_loss.data[0],
                'loss:': loss.data[0],
            })

            if (updates % config.eval_interval) in [0,1,2,3,4,5,6,7,8,9,10] :
                predicted_maps=multi_mask*x_input_map_multi
                # predicted_maps=Variable(feas_tgt)
                utils.bss_eval(config, predicted_maps,train_data['multi_spk_fea_list'], raw_tgt, train_data, dst='batch_outputwaddd')
                del predicted_maps,multi_mask,x_input_map_multi
                sdr_aver_batch=bss_test.cal('batch_outputwaddd/')
                lera.log({'SDR sample':sdr_aver_batch})
                SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
                print 'SDR_aver_now:',SDR_SUM.mean()


            total_loss += loss.data[0]
            report_total += num_total
            optim.step()
            if config.is_dis:
                optim_dis.step()

            updates += 1
            if updates%30==0:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss this batch: %6.3f,sgm loss: %6.6f,ss loss: %6.6f\n"
                        % (time.time()-start_time, epoch, updates, loss / num_total, total_loss_sgm/30.0, total_loss_ss/30.0))
                total_loss_sgm, total_loss_ss = 0, 0

            # continue

            if 0 or updates % config.eval_interval == 0 and epoch>1 :
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f\n"
                        % (time.time()-start_time, epoch, updates, total_loss / report_total))
                print('evaluating after %d updates...\r' % updates)
                # score = eval(epoch)
                for metric in config.metric:
                    scores[metric].append(score[metric])
                    lera.log({
                        'sgm_micro_f1': score[metric],
                    })
                    if metric == 'micro_f1' and score[metric] >= max(scores[metric]):
                        save_model(log_path+'best_'+metric+'_checkpoint.pt')
                    if metric == 'hamming_loss' and score[metric] <= min(scores[metric]):
                        save_model(log_path+'best_'+metric+'_checkpoint.pt')

                model.train()
                total_loss = 0
                start_time = 0
                report_total = 0

        except RuntimeError,eeee:
            print 'Erros here eeee: ',eeee
            continue
        except Exception,dddd:
            print '\n\n\nRare errors: ',dddd
            continue

        if updates % config.save_interval == 1:
            save_model(log_path+'sscn_v01b_{}.pt'.format(updates))


def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    e=epoch
    test_or_valid='valid'
    print 'Test or valid:',test_or_valid
    eval_data_gen=prepare_data('once',test_or_valid,config.MIN_MIX,config.MAX_MIX)
    # for raw_src, src, src_len, raw_tgt, tgt, tgt_len in validloader:
    SDR_SUM=np.array([])
    batch_idx=0
    global best_SDR,Var
    while True:
    # for ___ in range(2):
        print '-'*30
        eval_data=eval_data_gen.next()
        if eval_data==False:
            print 'SDR_aver_eval_epoch:',SDR_SUM.mean()
            break #如果这个epoch的生成器没有数据了，直接进入下一个epoch
        src = Variable(torch.from_numpy(eval_data['mix_feas']))

        raw_tgt = [sorted(spk.keys()) for spk in eval_data['multi_spk_fea_list']]
        feas_tgt=models.rank_feas(raw_tgt,eval_data['multi_spk_fea_list']) #这里是目标的图谱

        top_k=len(raw_tgt[0])
        # 要保证底下这几个都是longTensor(长整数）
        # tgt = Variable(torch.from_numpy(np.array([[0]+[dict_spk2idx[spk] for spk in spks]+[dict_spk2idx['<EOS>']] for spks in raw_tgt],dtype=np.int))).transpose(0,1) #转换成数字，然后前后加开始和结束符号。
        tgt = Variable(torch.ones(top_k+2,config.batch_size)) # 这里随便给一个tgt，为了测试阶段tgt的名字无所谓其实。

        src_len = Variable(torch.LongTensor(config.batch_size).zero_()+mix_speech_len).unsqueeze(0)
        tgt_len = Variable(torch.LongTensor([len(one_spk) for one_spk in eval_data['multi_spk_fea_list']])).unsqueeze(0)
        # tgt_len = Variable(torch.LongTensor(config.batch_size).zero_()+len(eval_data['multi_spk_fea_list'][0])).unsqueeze(0)
        if config.WFM:
            tmp_size=feas_tgt.size()
            assert len(tmp_size)==4
            feas_tgt_sum=torch.sum(feas_tgt,dim=1,keepdim=True)
            feas_tgt_sum_square=(feas_tgt_sum*feas_tgt_sum).expand(tmp_size)
            feas_tgt_square=feas_tgt*feas_tgt
            WFM_mask=feas_tgt_square/feas_tgt_sum_square

        if use_cuda:
            src = src.cuda().transpose(0,1)
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            feas_tgt = feas_tgt.cuda()
            if config.WFM:
                WFM_mask= WFM_mask.cuda()
        try:
            if 1 and len(opt.gpus) > 1:
                # samples, alignment = model.module.sample(src, src_len)
                samples, alignment, hiddens, predicted_masks = model.module.beam_sample(src, src_len, dict_spk2idx, tgt, beam_size=config.beam_size)
            else:
                samples, alignment, hiddens, predicted_masks = model.beam_sample(src, src_len, dict_spk2idx, tgt, beam_size=config.beam_size)
                # samples, alignment, hiddens, predicted_masks = model.beam_sample(src, src_len, dict_spk2idx, tgt, beam_size=config.beam_size)
        except TabError,info:
            print '**************Error occurs here************:', info
            continue

        if config.top1:
            predicted_masks=torch.cat([predicted_masks,1-predicted_masks],1)

        # '''
        # expand the raw mixed-features to topk_max channel.
        src = src.transpose(0,1)
        siz=src.size()
        assert len(siz)==3
        topk_max=feas_tgt.size()[1]
        assert samples[0][-1]==dict_spk2idx['<EOS>']
        topk_max=len(samples[0])-1
        x_input_map_multi=torch.unsqueeze(src,1).expand(siz[0],topk_max,siz[1],siz[2])
        if config.WFM:
            feas_tgt=x_input_map_multi.data*WFM_mask
        if 1 and len(opt.gpus) > 1:
            ss_loss = model.module.separation_loss(x_input_map_multi, predicted_masks, feas_tgt,Var)
        else:
            ss_loss = model.separation_loss(x_input_map_multi, predicted_masks, feas_tgt)
        print 'loss for ss,this batch:',ss_loss.data[0]
        lera.log({
            'ss_loss_'+test_or_valid: ss_loss.data[0],
        })

        del ss_loss,hiddens

        # '''''
        if batch_idx<=(500/config.batch_size): #only the former batches counts the SDR
            predicted_maps=predicted_masks*x_input_map_multi
            # predicted_maps=Variable(feas_tgt)
            utils.bss_eval2(config, predicted_maps,eval_data['multi_spk_fea_list'], raw_tgt, eval_data, dst='batch_outputwaddd')
            del predicted_maps,predicted_masks,x_input_map_multi
            SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_outputwaddd/'))
            print 'SDR_aver_now:',SDR_SUM.mean()
            lera.log({'SDR sample':SDR_SUM.mean()})
            # raw_input('Press any key to continue......')
        elif batch_idx==(500/config.batch_size)+1 and SDR_SUM.mean()>best_SDR: #only record the best SDR once.
            print 'Best SDR from {}---->{}'.format(best_SDR,SDR_SUM.mean())
            best_SDR=SDR_SUM.mean()
            # save_model(log_path+'checkpoint_bestSDR{}.pt'.format(best_SDR))

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
            train(i)
        else:
            eval(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
