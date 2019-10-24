#coding=utf8
import os
import numpy as np
import random
import re
import soundfile as sf
import resampy
import librosa
import argparse
import data.utils as utils

# Add the config.
parser = argparse.ArgumentParser(description='predata scripts.')
parser.add_argument('-config', default='config.yaml', type=str,
# parser.add_argument('-config', default='config_mix.yaml', type=str,
                    help="config file")
opt = parser.parse_args()
config = utils.read_config(opt.config)

config.MAX_LEN=40000
channel_first=config.channel_first
np.random.seed(1)#设定种子
random.seed(1)

aim_path='fake/fake/DL4SS_Keras/Torch_multi/Dataset_Multi/1/'+config.DATASET
# 训练文件列表
TRAIN_LIST = aim_path+'/train_list'
# 验证文件列表
VALID_LIST = aim_path+'/valid_list'
# 测试文件列表
TEST_LIST = aim_path+'/test_list'
# 未登录文件列表
UNK_LIST = aim_path+'/unk_list'

def split_forTrainDevTest(spk_list,train_or_test):
    '''为了保证一个统一的训练和测试的划分标准，不得不用通用的一些方法来限定一下,
    这里采用的是用sorted先固定方法的排序，那么不论方法或者seed怎么设置，训练测试的划分标准维持不变，
    也就是数据集会维持一直'''
    length=len(spk_list)
    # spk_list=sorted(spk_list,key=lambda x:(x[1]))#这个意思是按照文件名的第二个字符排序
    # spk_list=sorted(spk_list)#这个意思是按照文件名的第1个字符排序,暂时采用这种
    spk_list=sorted(spk_list,key=lambda x:(x[-1]))#这个意思是按照文件名的最后一个字符排序
    #TODO:暂时用最后一个字符排序，这个容易造成问题，可能第一个比较不一样的，这个需要注意一下
    if train_or_test=='train':
        return spk_list[:int(round(0.7*length))]
    elif train_or_test=='valid':
        return spk_list[(int(round(0.7*length))+1):int(round(0.8*length))]
    elif train_or_test=='test':
        return spk_list[(int(round(0.8*length))+1):]
    else:
        raise ValueError('Wrong input of train_or_test.')

def prepare_datasize(gen):
    data=gen.next()
    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    #暂时输出的是：语音长度、语音频率数量、视频截断之后的长度
    print 'datasize:',data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])
    return data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])

def prepare_data(mode,train_or_test,min=None,max=None,given_wav=''):
    '''
    :param
    mode: type str, 'global' or 'once' ， global用来获取全局的spk_to_idx的字典，所有说话人的列表等等
    train_or_test:type str, 'train','valid' or 'test'
     其中把每个文件夹每个人的按文件名的排序的前70%作为训练，70-80%作为valid，最后20%作为测试
    :return:
    '''
    # 如错有预订的min和max，主要是为了主程序做valid的时候统一某个固定的说话人的个数上
    if min:
        config.MIN_MIX=min
    if max:
        config.MAX_MIX=max
    if given_wav:
        wav_mix0, rate = sf.read(given_wav)  # wav_mix 是采样值，rate 是采样频率
        if rate != config.FRAME_RATE:
            # 如果频率不是设定的频率则需要进行转换
            wav_mix0 = resampy.resample(wav_mix0, rate, config.FRAME_RATE, filter='kaiser_best')
        wav_length=len(wav_mix0)
        config.MAX_LEN=wav_length

    mix_speechs=np.zeros((config.batch_size,config.MAX_LEN))
    mix_feas=[]#应该是bs,n_frames,n_fre这么多
    mix_phase=[]#应该是bs,n_frames,n_fre这么多
    aim_fea=[]#应该是bs,n_frames,n_fre这么多
    aim_spkid=[] #np.zeros(config.batch_size)
    aim_spkname=[] #np.zeros(config.batch_size)
    query=[]#应该是batch_size，shape(query)的形式，用list再转换把
    multi_spk_fea_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典
    multi_spk_wav_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典

    #目标数据集的总data，底下应该存放分目录的文件夹，每个文件夹应该名字是sX
    data_path=aim_path+'/data'
    #语音刺激
    if config.MODE==1:
        if config.DATASET=='WSJ0': #开始构建数据集
            batch_idx=0
            all_samples_list={}
            sample_idx={}
            batch_mix={}
            mix_number_list=range(config.MIN_MIX,config.MAX_MIX+1)
            number_samples_all=1
            for ___ in range(number_samples_all):
                mix_len=0
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

                # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
                # wav_mix, rate = sf.read('/data3/shijing/朗读-单通道语音.wav')  # wav_mix 是采样值，rate 是采样频率
                wav_mix, rate = sf.read(given_wav)  # wav_mix 是采样值，rate 是采样频率
                if len(wav_mix.shape) > 1:
                    wav_mix = wav_mix[:, 0]
                if rate != config.FRAME_RATE:
                    # 如果频率不是设定的频率则需要进行转换
                    wav_mix = resampy.resample(wav_mix, rate, config.FRAME_RATE, filter='kaiser_best')
                if wav_mix.shape[0] > config.MAX_LEN:  # 根据最大长度裁剪
                    wav_mix = wav_mix[:config.MAX_LEN]
                # 更新混叠语音长度
                if wav_mix.shape[0] > mix_len:
                    mix_len = wav_mix.shape[0]

                wav_mix -= np.mean(wav_mix)  # 语音信号预处理，先减去均值
                wav_mix /= np.max(np.abs(wav_mix))  # 波形幅值预处理，幅值归一化
                
                if config.IS_LOG_SPECTRAL:
                    feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                        config.FRAME_SHIFT,
                                                                                        window=config.WINDOWS)))
                                         + np.spacing(1))
                else:
                    feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                     config.FRAME_SHIFT,)))

                mix_speechs[batch_idx,:]=wav_mix
                mix_feas.append(feature_mix)
                mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                     config.FRAME_SHIFT,)))
                batch_idx+=1
                # print 'batch_dix:{}/{},'.format(batch_idx,config.batch_size),
                if batch_idx==config.batch_size: #填满了一个batch
                    #下一个batch的混合说话人个数， 先调整一下
                    mix_k=random.sample(mix_number_list,1)[0]
                    mix_feas=np.array(mix_feas)
                    mix_phase=np.array(mix_phase)
                    aim_fea=np.array(aim_fea)
                    # aim_spkid=np.array(aim_spkid)
                    query=np.array(query)
                    print 'spk_list_from_this_gen:{}'.format(aim_spkname)
                    print 'aim spk list:', [one.keys() for one in multi_spk_fea_list]
                    # print '\nmix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkname.shape,query.shape,all_spk_num:'
                    # print mix_speechs.shape,mix_feas.shape,aim_fea.shape,len(aim_spkname),query.shape,len(all_spk)
                    if mode=='global':
                        return None
                        # return all_spk,dict_spk_to_idx,dict_idx_to_spk,\
                        #       aim_fea.shape[1],aim_fea.shape[2],32,len(all_spk),batch_total
                              #上面的是：语音长度、语音频率、视频分割多少帧 TODO:后面把这个替换了query.shape[1]
                    elif mode=='once':
                        return {'mix_wav':mix_speechs,
                               'mix_feas':mix_feas,
                               'mix_phase':mix_phase,
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
                sample_idx[mix_k]+=1

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
