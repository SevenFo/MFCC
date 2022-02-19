from numpy.lib.index_tricks import nd_grid
import torch
import torchaudio
import numpy as np

import torchaudio.transforms as tat
import librosa
waveform,sr = torchaudio.load("barking_recorded.wav")
waveform = waveform[0]

waveform = torch.tensor(waveform,dtype=torch.float32)

spec_torch = tat.Spectrogram(n_fft=2048,center=True,win_length=2048,hop_length=1024,pad=0,power=2)(waveform)


spec_torch
stft_librosa = librosa.stft(y=waveform.numpy(),
                            hop_length=1024,
                            center = True,
                            pad_mode="reflect",
                            n_fft=2048)
spec_librosa = pow(np.abs(stft_librosa),2)
print(spec_librosa.shape,spec_torch.shape)




mel_scale = tat.MelScale(n_mels=128,sample_rate=sr,f_min=0,f_max=sr//2,n_stft=1025)
mel_spce_torch = mel_scale(spec_torch)
mel_spce_torch_t = torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_fft=2048,win_length=2048,hop_length=1024,f_max=sr//2)(waveform)

mel_spce_librosa = librosa.feature.melspectrogram(waveform.numpy(),sr=sr,n_fft=2048,hop_length=1024,win_length=2048,pad_mode='reflect',center=True)
mel_filter = librosa.filters.mel(sr=sr,n_fft=2048,n_mels=128,fmax=sr//2,htk=True,norm=None)
mel_filter = np.transpose(np.array(mel_filter))
mel_spce_librosa_2 = np.transpose(np.matmul(np.transpose(spec_librosa.reshape(1025,174)),mel_filter))

# mel_spce_librosa_2 = 

pass