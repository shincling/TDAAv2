#coding=utf8
import numpy as np
import os
import soundfile as sf
from separation import bss_eval_sources
'''
目标文件夹path里的文件格式如下：
Notes: xxx 可以是任意字符(不用保持一致），一般用来指示说话人或者样本的标签
以下情况以总计k段混合语音，2个人混合的语音为例，3个人或多个人的以此类推

第1段混合语音：
0_xxx_real.wav   # 混合语音中的第一路说话人的纯净语音 (顺序无所谓) 
0_xxx_real.wav   # 混合语音中的第二路说话人的纯净语音 (顺序无所谓)
0_xxx_pre.wav    # 混合语音中预测的某一路语音(顺序无所谓)
0_xxx_pre.wav    # 混合语音中预测的另一路语音(顺序无所谓)

第2段混合语音：
1_xxx_real.wav    
1_xxx_real.wav    
1_xxx_pre.wav
1_xxx_pre.wav

第3段混合语音：
2_xxx_real.wav    
2_xxx_real.wav    
2_xxx_pre.wav
2_xxx_pre.wav

......

第k段混合语音：
k_xxx_real.wav
k_xxx_real.wav
k_xxx_pre.wav
k_xxx_pre.wav

'''

path='batch_output/'
# path='/home/sw/Shin/Codes/DL4SS_Keras/TDAA_beta/batch_output2/'
def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    if src_ref.shape[0]==1:
        src_anchor=src_anchor[0]
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def cal(path,tmp=None):
    mix_number=len(set([l.split('_')[0] for l in os.listdir(path) if l[-3:]=='wav']))
    print(('num of mixed :',mix_number))
    SDR_sum=np.array([])
    SDRi_sum=np.array([])
    for idx in range(mix_number):
        pre_speech_channel=[]
        aim_speech_channel=[]
        mix_speech=[]
        for l in sorted(os.listdir(path)):
            if l[-3:]!='wav':
                continue
            if l.split('_')[0]==str(idx):
                if 'True_mix' in l:
                    mix_speech.append(sf.read(path+l)[0])
                if 'real' in l and 'noise' not in l:
                    aim_speech_channel.append(sf.read(path+l)[0])
                if 'pre' in l:
                    pre_speech_channel.append(sf.read(path+l)[0])

        assert len(aim_speech_channel)==len(pre_speech_channel)
        aim_speech_channel=np.array(aim_speech_channel)
        pre_speech_channel=np.array(pre_speech_channel)
        mix_speech=np.array(mix_speech)
        assert mix_speech.shape[0]==1
        mix_speech=mix_speech[0]

        # print aim_speech_channel.shape
        # print pre_speech_channel.shape

        # print('aim SDR:',aim_speech_channel[:,16000:16005])
        # print('pre SDR:',pre_speech_channel[:,16000:16005])
        result=bss_eval_sources(aim_speech_channel,pre_speech_channel)
        print(('SDR',result))
        SDR_sum=np.append(SDR_sum,result[0])

        # result=bss_eval_sources(aim_speech_channel,aim_speech_channel)
        # result_sdri=cal_SDRi(aim_speech_channel,pre_speech_channel,mix_speech)
        # print 'SDRi:',result_sdri
        result_sdri=cal_SISNRi(aim_speech_channel,pre_speech_channel[result[-1]],mix_speech)
        print(('SI-SNR',result_sdri))
        # for ii in range(aim_speech_channel.shape[0]):
        #     result=cal_SISNRi(aim_speech_channel[ii],pre_speech_channel[ii],mix_speech[ii])
        #     print('SI-SNR',result)
        SDRi_sum=np.append(SDRi_sum,result_sdri)

    print(('SDR_Aver for this batch:',SDR_sum.mean()))
    # print 'SDRi_Aver for this batch:',SDRi_sum.mean()
    return SDR_sum.mean(),SDRi_sum.mean()

# cal(path)

