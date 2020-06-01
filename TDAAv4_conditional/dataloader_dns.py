#coding=utf8

import json
import math
import os

import numpy as np
import torch
import torch.utils.data as data

import librosa

np.random.seed(1)  # 设定种子

def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    sample_rate, segment_len, __= batch[0][-3:]
    mix_infos,s1_infos,s2_infos=[],[],[]
    for ii in batch:
        mix_infos.append(ii[0])
        s1_infos.append(ii[1])
        s2_infos.append(ii[2])

    mix_path_list = []
    s1_path_list = []
    s2_path_list = []
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info
        mix_path_list.append(mix_path)
        s1_path = s1_info
        s1_path_list.append(s1_path)
        s2_path = s2_info
        s2_path_list.append(s2_path)
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate) # 480000
        s1, _ = librosa.load(s1_path, sr=sample_rate) # 480000
        s2, _ = librosa.load(s2_path, sr=sample_rate) # 480000
        # merge s1 and s2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2 480000*2
        utt_len = mix.shape[-1]
        if segment_len >= 0: # 480000
            # segment 这里应该是想把太长的语音切成片端，分批次处理完，解决显存问题
            for i in range(0, utt_len - segment_len + 1, segment_len): # 0，1，480000 只一次
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0: # 480000/480000 == 0
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    wav_name_list = [mix_path_list, s1_path_list, s2_path_list]
    return mixtures, sources, wav_name_list


def load_mixtures_and_sources_test(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    sample_rate = batch[0][-1]
    mix_infos,s1_infos=[],[]
    for ii in batch:
        mix_infos.append(ii[0])
        s1_infos.append(ii[1])

    mix_path_list = []
    s1_path_list = []
    # for each utterance
    for mix_info, s1_info in zip(mix_infos, s1_infos):
        mix_path = mix_info
        mix_path_list.append(mix_path)
        s1_path = s1_info
        s1_path_list.append(s1_path)
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate) # 480000
        s1, _ = librosa.load(s1_path, sr=sample_rate) # 480000
        # full utterance
        mixtures.append(mix)
        sources.append(s1)
    wav_name_list = [mix_path_list, s1_path_list]
    return mixtures, sources, wav_name_list


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

class DNSDataset_train(data.Dataset):
    def __init__(self,path_root,sample_rate,speech_length,chunk_length):
        self.path_root=path_root
        self.sample_rate=sample_rate
        self.speech_length=speech_length
        self.chunk_length=chunk_length # 似乎没什么作用?
        self.clean_files_list = sorted(os.listdir(path_root + '/' + 'clean'), key=lambda x: int(x.split('_')[-1][:-4]))
        self.mixture_files_list = sorted(os.listdir(path_root + '/' + 'mix'), key=lambda x: int(x.split('_')[-1][:-4]))
        self.noise_files_list = sorted(os.listdir(path_root + '/' + 'noise'), key=lambda x: int(x.split('_')[-1][:-4]))

        assert len(self.clean_files_list)==len(self.mixture_files_list)==len(self.noise_files_list)==60000
        self.len = 60000

    def __getitem__(self, index):
        clean_path=self.path_root+'/clean/'+self.clean_files_list[index] # 确定一下这里sorted后一一对应,用编号排序的，必然对
        mixture_path=self.path_root+'/mix/'+self.mixture_files_list[index]
        noise_path=self.path_root+'/noise/'+self.noise_files_list[index]
        return mixture_path,clean_path,noise_path,self.sample_rate,self.speech_length,self.chunk_length

    def __len__(self):
        return self.len


class DNSLoader(data.DataLoader):

    def __init__(self, *args, **kwargs):
        super(DNSLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list of paths. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # clean_path,mixture_path,noise_path,sample_rate,speech_length = batch

    # batch should be located in list
    mixtures, sources, wav_name_list = load_mixtures_and_sources(batch)
    len_limit=batch[0][-1] # chunk的最后一位用来确定生成的语音片段长度
    if len_limit>0: # get the random range of the given mixture
        range_low=np.random.randint(0,mixtures[0].shape[0]-len_limit)
        range_high=range_low+len_limit
        mixtures=[mix[range_low:range_high] for mix in mixtures]
        sources=[source[range_low:range_high] for source in sources]

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures]) # 统一都是16000

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous() # bs*steps*spk -> bs*spk*steps

    return mixtures_pad, ilens, sources_pad[:,0], sources_pad[:,1], wav_name_list


class DNSDataset_test(data.Dataset):
    def __init__(self,path_root,sample_rate):
        self.path_root=path_root
        self.sample_rate=sample_rate
        self.clean_files_list = sorted(os.listdir(path_root + '/' + 'clean'), key=lambda x: int(x.split('_')[-1][:-4]))
        self.mixture_files_list = sorted(os.listdir(path_root + '/' + 'noisy'), key=lambda x: int(x.split('_')[-1][:-4]))

        assert len(self.clean_files_list)==len(self.mixture_files_list)==150
        self.len = 150

    def __getitem__(self, index):
        clean_path=self.path_root+'/clean/'+self.clean_files_list[index] # 确定一下这里sorted后一一对应,用编号排序的，必然对
        mixture_path=self.path_root+'/noisy/'+self.mixture_files_list[index]
        return mixture_path,clean_path,self.sample_rate

    def __len__(self):
        return self.len


class DNSLoader_test(data.DataLoader):

    def __init__(self, *args, **kwargs):
        super(DNSLoader_test, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_test


def _collate_fn_test(batch):
    """
    Args:
        batch: list of paths. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # clean_path,mixture_path,noise_path,sample_rate,speech_length = batch

    # batch should be located in list
    mixtures, sources, wav_name_list = load_mixtures_and_sources_test(batch)

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures]) # 统一都是16000

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value) # 这里的padding?
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    # sources_pad = sources_pad.permute((0, 2, 1)).contiguous() # bs*steps*spk -> bs*spk*steps
    sources_pad = sources_pad.unsqueeze(1).contiguous() # bs*steps*spk -> bs*spk*steps

    return mixtures_pad, ilens, sources_pad[:,0:], wav_name_list


def main_load(root_path,sample_rate,speech_length):
    dataloader=DNSLoader(
        dataset=DNSDataset_train(root_path,sample_rate,speech_length,80000),
        batch_size=6,
        shuffle=False,
        num_workers=4
    )
    return dataloader

if __name__=="__main__":
    path='/data1/shijing_data/clone/DNS-Challenge/training'
    sample_rate=16000
    length=30
    main_generator=main_load(path,sample_rate,sample_rate*length) # 30是谁定的?
    for idx,batch in enumerate(main_generator):
        mixtures, lens, sources,noise, spk_list = batch
        print(idx)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        print(noise.size())
        print(spk_list[:5],spk_list[-5:])
        # if idx < 10:
        #     print(mixtures)
        #     print(sources)
        # else:
        #     break

