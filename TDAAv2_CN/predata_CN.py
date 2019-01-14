#coding=utf8
import os
import numpy as np
import random
import re
import soundfile as sf
import resampy
import librosa
import argparse
import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))

# Add the config.
parser = argparse.ArgumentParser(description='predata scripts.')
parser.add_argument('-config', default='config_CN.yaml', type=str,
                    help="config file")
opt = parser.parse_args()
config = read_config(opt.config)

channel_first=config.channel_first
np.random.seed(1)#设定种子
random.seed(1)

aim_path='/data3/data_aishell/wav/'  #400 in total
noise_path='/data3/noise/'  #400 in total
# 训练文件列表
TRAIN_LIST = aim_path+'/train_list'
# 验证文件列表
VALID_LIST = aim_path+'/valid_list'
# 测试文件列表
TEST_LIST = aim_path+'/test_list'
# 未登录文件列表
UNK_LIST = aim_path+'/unk_list'

def split_forTrainDevTest(given_list,train_or_test,phase):
    '''为了保证一个统一的训练和测试的划分标准，用通用的一些方法来限定一下,
    数据一共有400个说话人，其中 train/eval占90%， test与他们不重合，占10% （划分依据是说话人ID的最后一个数字，如果是0,则是test
    对于 train 和　eval，　按照每个说话人里面的语音片段最后一个字符来划分，８和９结束的为eval，否则为train

    phase参数用来决定是说话人的划分，　train/eval　还是　 test，
    　　　　　或者是train/eval里的具体音频片段的划分成　train 还是 eval　
    '''
    length=len(given_list)
    #TODO:暂时用最后一个字符decide，这个容易造成问题，可能第一个比较不一样的，这个需要注意一下
    if phase=='spks': #用来划分说话人是 train/valid 的 还是test的（不重合）
        if train_or_test=='train' or train_or_test=='valid':
            output_list=[name for name in given_list if name[-1]!='0'] #
        else:
            output_list=[name for name in given_list if name[-1]=='0'] # test spks covers 10% which not overlaped with train or valid.

    elif phase=='wavs': #用来划分训练测试说话人里的没一条语音 train还是valid的语音
        if train_or_test=='train':
            output_list=[wav for wav in given_list if int(wav[-5]) in range(8)] # test spks covers 10% which not overlaped with train or valid.
        elif train_or_test=='valid':
            output_list=[wav for wav in given_list if wav[-5]=='8' or  wav[-5]=='9'] #
        else:
            raise ValueError('Wrong input of train or test in split wavs.')

    else:
        raise ValueError('Wrong input of phase.')

    return output_list

def process_signal(signal,rate,aim_len,num_ordier=None):
    "输入语音wav信号，用来预处理合适的长度和采样率，已经确保是大于config.MIN_LEN了"
    if len(signal.shape) > 1:
        signal = signal[:, 0]

    if rate != config.FRAME_RATE: # 先采样到设定的采样率上
        # 如果频率不是设定的频率则需要进行转换
        signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')

    if signal.shape[0] > aim_len:  # 根据最大长度裁剪
        signal = signal[:aim_len]

    if num_ordier == 0 and signal.shape[0] < aim_len:  # 第一个信号根据最大长度用 0 补齐,
        signal=np.append(signal,np.zeros(aim_len - signal.shape[0]))

    if num_ordier != 0 and signal.shape[0] < aim_len:  # 之后的信号引入一个偏移，然后用根据最大长度用 0 补齐,
        shift_frames=random.sample(range(aim_len - signal.shape[0]+1),1)[0] # 0~长度差
        signal=np.append(np.append(np.zeros(shift_frames),signal),np.zeros(aim_len - signal.shape[0] - shift_frames))

    if config.normalize: #如果需要归一化的话
        signal -= np.mean(signal)  # 语音信号预处理，先减去均值
        signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

    return signal


def prepare_data(mode,train_or_test,min=None,max=None,add_noise=True):
    '''
    :param
    mode: type str, 'global' or 'once' ， global用来获取全局的spk_to_idx的字典，所有说话人的列表等等
    train_or_test:type str, 'train','valid' or 'test'
     其中把每个文件夹每个人的按文件名的排序的前70%作为训练，70-80%作为valid (overlapped)，最后20%作为测试(unknonw spks)
    :return: generator of dataset.
    '''
    # 如错有预订的min和max，主要是为了主程序做valid的时候统一某个固定的说话人的个数上
    if min:
        config.MIN_MIX=min
    if max:
        config.MAX_MIX=max

    mix_speechs=np.zeros((config.batch_size,config.MAX_LEN))
    mix_feas=[]#应该是bs,n_frames,n_fre这么多
    mix_phase=[]#应该是bs,n_frames,n_fre这么多
    mix_phase=[]#应该是bs,n_frames,n_fre这么多
    aim_fea=[]#应该是bs,n_frames,n_fre这么多
    aim_spkid=[] #np.zeros(config.batch_size)
    aim_spkname=[] #np.zeros(config.batch_size)
    query=[]#应该是batch_size，shape(query)的形式，用list再转换把
    multi_spk_fea_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典
    multi_spk_wav_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典

    #目标数据集的总data，底下应该存放分目录的文件夹，每个文件夹应该名字是sX
    data_path=aim_path

    #语音刺激
    if config.MODE==1:
        if config.DATASET=='CN': #开始构建数据集
            all_spk=sorted(os.listdir(data_path))
            batch_idx=0
            batch_total=config.num_samples_one_epoch/config.batch_size
            print 'batch_total_num:',batch_total
            number_samples_all=config.num_samples_one_epoch

            mix_number_list=range(config.MIN_MIX,config.MAX_MIX+1)
            for ___ in range(number_samples_all): # 每一步合成一条语音
                if ___==number_samples_all-1:
                    print 'This epoch ends here.'
                    yield False

                mix_len=0
                mix_k=random.sample(mix_number_list,1)[0] # 这一条需要几个说话人
                aim_spk_lists=split_forTrainDevTest(all_spk,train_or_test,phase='spks') # find the spks with given train/valid/ or test
                aim_spk_k=random.sample(aim_spk_lists,mix_k) # 这一条采样出来的说话人

                multi_fea_dict_this_sample={}
                multi_wav_dict_this_sample={}
                multi_name_list_this_sample=[]
                multi_db_dict_this_sample={}

                if config.dB and config.MIN_MIX==config.MAX_MIX==2:
                    dB_rate=10**(config.dB/20.0*np.random.rand())#e**(0——0.5)
                    print 'channel to change with dB:',dB_rate

                for k,spk in enumerate(aim_spk_k): #对于每个采样出来的说话人的一条
                    path_this_spk=data_path+spk+'/'
                    wavs_this_spk=os.listdir(path_this_spk)
                    aim_wavs=split_forTrainDevTest(wavs_this_spk,train_or_test,phase='wavs') # find the spks with given train/valid/ or test

                    while True: #确保抽样出长度大于min_len
                        sampled_wav_name=random.sample(aim_wavs,1)[0]
                        spk_speech_path=path_this_spk+sampled_wav_name
                        signal, rate = sf.read(spk_speech_path)  # signal 是采样值，rate 是采样频率
                        if signal.shape[0] < config.MIN_MIX:  # 根据最大长度裁剪
                            continue
                        else:
                            break

                    signal = process_signal(signal,rate, config.MAX_LEN, num_ordier=k)

                    if k==0:#第一个作为目标
                        # ratio=10**(aim_spk_db_k[k]/20.0)
                        # signal=ratio*signal
                        aim_spkname.append(aim_spk_k[0])
                        aim_spkid.append(aim_spkname)
                        wav_mix=signal
                        if not config.IS_LOG_SPECTRAL: #暂时都不用log
                            aim_fea_clean = np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH, config.FRAME_SHIFT)))
                        aim_fea.append(aim_fea_clean)
                        # 把第一个人顺便也注册进去混合dict里
                        multi_fea_dict_this_sample[spk]=aim_fea_clean
                        multi_wav_dict_this_sample[spk]=signal

                    else:
                        # ratio=10**(aim_spk_db_k[k]/20.0)
                        # signal=ratio*signal
                        wav_mix = wav_mix + signal  # 混叠后的语音
                        if not config.IS_LOG_SPECTRAL:
                            some_fea_clean = np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH, config.FRAME_SHIFT,)))
                        multi_fea_dict_this_sample[spk]=some_fea_clean
                        multi_wav_dict_this_sample[spk]=signal

                multi_spk_fea_list.append(multi_fea_dict_this_sample) #把这个sample的dict传进去
                multi_spk_wav_list.append(multi_wav_dict_this_sample) #把这个sample的dict传进去

                assert wav_mix.shape[0]==config.MAX_LEN
                wav_mix=process_signal(wav_mix,rate,config.MAX_LEN)
                # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
                if config.IS_LOG_SPECTRAL:
                    feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH, config.FRAME_SHIFT,))) + np.spacing(1))
                else:
                    feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH, config.FRAME_SHIFT,)))

                mix_speechs[batch_idx,:]=wav_mix
                mix_feas.append(feature_mix)
                mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH, config.FRAME_SHIFT,)))
                batch_idx+=1
                print 'batch_dix:{}/{},'.format(batch_idx,config.batch_size),

                if batch_idx==config.batch_size: #填满了一个batch
                    #下一个batch的混合说话人个数， 先调整一下
                    mix_feas=np.array(mix_feas)
                    mix_phase=np.array(mix_phase)
                    aim_fea=np.array(aim_fea)
                    query=np.array(query)
                    print 'spk_list_from_this_gen:{}'.format(aim_spkname)
                    print 'aim spk list:', [one.keys() for one in multi_spk_fea_list]
                    # print '\nmix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkname.shape,query.shape,all_spk_num:'
                    # print mix_speechs.shape,mix_feas.shape,aim_fea.shape,len(aim_spkname),query.shape,len(all_spk)
                    if mode=='global':
                        all_spk=sorted(all_spk)
                        all_spk.insert(0,'<BOS>') #添加两个结构符号，来标识开始或结束。
                        all_spk.append('<EOS>')
                        all_spk_eval=sorted(all_spk_eval)
                        all_spk_test=sorted(all_spk_test)
                        dict_spk_to_idx={spk:idx for idx,spk in enumerate(all_spk)}
                        dict_idx_to_spk={idx:spk for idx,spk in enumerate(all_spk)}
                        yield all_spk,dict_spk_to_idx,dict_idx_to_spk,\
                              aim_fea.shape[1],aim_fea.shape[2],32,len(all_spk),batch_total
                              #上面的是：语音长度、语音频率、视频分割多少帧 TODO:后面把这个替换了query.shape[1]
                    elif mode=='once':
                        yield {'mix_wav':mix_speechs,
                               'mix_feas':mix_feas,
                               'mix_phase':mix_phase,
                               'aim_fea':aim_fea,
                               'aim_spkname':aim_spkname,
                               'query':query,
                               'num_all_spk':len(all_spk),
                               'multi_spk_fea_list':multi_spk_fea_list,
                               'multi_spk_wav_list':multi_spk_wav_list,
                               'batch_total':batch_total,
                               }

                    batch_idx=0
                    mix_speechs=np.zeros((config.batch_size,config.MAX_LEN))
                    mix_feas=[]#应该是bs,n_frames,n_fre这么多
                    mix_phase=[]
                    aim_fea=[]#应该是bs,n_frames,n_fre这么多
                    aim_spkid=[] #np.zeros(config.batch_size)
                    aim_spkname=[]
                    query=[]#应该是batch_size，shape(query)的形式，用list再转换把
                    multi_spk_fea_list=[]
                    multi_spk_wav_list=[]

        else:
            raise ValueError('No such dataset:{} for Speech.'.format(config.DATASET))
        pass

    #图像刺激
    elif config.MODE==2:
        pass

    #视频刺激
    elif config.MODE==3:
        raise ValueError('No such dataset:{} for Video'.format(config.DATASET))
    #概念刺激
    elif config.MODE==4:
        pass

    else:
        raise ValueError('No such Model:{}'.format(config.MODE))

print 'hh'
cc=prepare_data('once','train')
cc.next()
pass
