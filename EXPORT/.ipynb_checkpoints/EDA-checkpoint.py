##################################################
# wave, spectrum, mel spectogram, MFCC plot 함수 #
##################################################

import librosa
import librosa.display as dsp
from IPython.display import Audio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# audio path?
def fourier_transform(path_wave):
    data,sample_rate = librosa.load(path_wave)
    fft = np.fft.fft(data) 
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sample_rate, len(magnitude))
    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(magnitude)/2)]

    plt.plot(left_frequency, left_magnitude)

    plt.title('Fourier transform')
    plt.show()
    
    return fft

def get_audio(path_wave):

    # Get Audio from the location
    data,sample_rate = librosa.load(path_wave)

    # Plot the audio wave
    plt.plot(data)
    plt.title('Waveform')
    plt.show()

    # Show the widget
    return Audio(data=data,rate=sample_rate)


def RMS_wave(path_wave):

    data,sample_rate = librosa.load(path_wave)

    rms = librosa.feature.rms(y=data)


    times = librosa.times_like(rms)

    plt.plot(times, rms[0])
    plt.title("RMS")
    plt.show()
    
    return rms

def Mel_s(path_wave, frame_length = 0.025, frame_stride = 0.010):

    data,sample_rate = librosa.load(path_wave)

    input_nfft = int(round(sample_rate*frame_length))
    input_stride = int(round(sample_rate*frame_stride))

    S = librosa.feature.melspectrogram(y=data, n_mels=100, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(data)/sample_rate, np.shape(S)))

    S_dB  = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize = (14,5))
    librosa.display.specshow(S_dB, 
                             sr=sample_rate, 
                             hop_length=input_stride,
                             x_axis='time',
                             y_axis='log')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Sepctogram")
    
    return S_dB


def mfcc(path_wave, frame_length = 0.025, frame_stride = 0.010):

    data,sample_rate = librosa.load(path_wave)

    input_nfft = int(round(sample_rate*frame_length))
    input_stride = int(round(sample_rate*frame_stride))

    S = librosa.feature.melspectrogram(y=data, n_mels=100, n_fft=input_nfft, hop_length=input_stride)
    S_dB  = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=S_dB, n_mfcc=20)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(delta2_mfcc)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()
    
    return delta2_mfcc