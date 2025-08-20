import re
import traceback
import librosa
import sklearn
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore")

#feature extraction   
def ext(file_data):
    path='./uploads/'
    n_songs = len(file_data)
    id = 0
    feature_set = pd.DataFrame()
    songname_vector = pd.Series()
    tempo_vector = pd.Series()
    duration_vector = pd.Series()
    rms_mean = pd.Series()
    rms_var = pd.Series()
    zcr_mean = pd.Series()
    zcr_var = pd.Series()
    cent_mean = pd.Series()
    cent_var = pd.Series()
    spec_bw_mean = pd.Series()
    spec_bw_var = pd.Series()
    rolloff_mean = pd.Series()
    rolloff_var = pd.Series()
    frame_mean = pd.Series()
    frame_var = pd.Series()
    chroma_stft_mean = np.zeros([n_songs,12])
    chroma_stft_var = np.zeros([n_songs,12])
    chroma_cq_mean = np.zeros([n_songs,12])
    chroma_cq_var = np.zeros([n_songs,12])
    chroma_cens_mean = np.zeros([n_songs,12])
    chroma_cens_var = np.zeros([n_songs,12])
    mel_mean = np.zeros([n_songs,128])
    mel_var = np.zeros([n_songs,128])
    mfcc_mean = np.zeros([n_songs,20])
    mfcc_var = np.zeros([n_songs,20])
    mfcc_delta_mean = np.zeros([n_songs,20])
    mfcc_delta_var = np.zeros([n_songs,20])
    contrast_mean = np.zeros([n_songs,7])
    contrast_var = np.zeros([n_songs,7])
    poly_mean = np.zeros([n_songs,3])
    poly_var = np.zeros([n_songs,3])
    tonnetz_mean = np.zeros([n_songs,6])
    tonnetz_var = np.zeros([n_songs,6])
    harm_mean = np.zeros([n_songs,12])
    harm_var = np.zeros([n_songs,12])
    perc_mean = np.zeros([n_songs,12])
    perc_var = np.zeros([n_songs,12])
    for file_name in file_data:
        try:
            songname = path + file_name
            y, sr = librosa.load(songname, duration=60)
            S = np.abs(librosa.stft(y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
            melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_delta = librosa.feature.delta(mfcc)
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            poly_features = librosa.feature.poly_features(S=S, sr=sr, order=2)
            harmonic = librosa.effects.harmonic(y)
            percussive = librosa.effects.percussive(y)
            harm_chroma_stft = librosa.feature.chroma_stft(y=harmonic, sr=sr)
            perc_chroma_stft = librosa.feature.chroma_stft(y=percussive, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            songname_vector.at[id] = file_name 
            tempo_vector.at[id] = tempo[0]
            duration_vector.at[id] = librosa.get_duration(filename=songname) 
            rms_mean.at[id] = np.mean(rms) 
            rms_var.at[id] = np.var(rms)
            zcr_mean.at[id] = np.mean(zcr) 
            zcr_var.at[id] = np.var(zcr)
            cent_mean.at[id] = np.mean(cent) 
            cent_var.at[id] = np.var(cent)
            spec_bw_mean.at[id] = np.mean(spec_bw) 
            spec_bw_var.at[id] = np.var(spec_bw)
            rolloff_mean.at[id] = np.mean(rolloff) 
            rolloff_var.at[id] = np.var(rolloff)
            frame_mean.at[id] = np.mean(frames_to_time) 
            frame_var.at[id] = np.var(frames_to_time)
            chroma_stft_mean[id] = np.mean(chroma_stft, axis=1) 
            chroma_stft_var[id] = np.var(chroma_stft, axis=1)
            chroma_cq_mean[id] = np.mean(chroma_cq, axis=1) 
            chroma_cq_var[id] = np.var(chroma_cq, axis=1)
            chroma_cens_mean[id] = np.mean(chroma_cens, axis=1) 
            chroma_cens_var[id] = np.var(chroma_cens, axis=1)
            mel_mean[id] = np.mean(melspectrogram, axis=1) 
            mel_var[id] = np.var(melspectrogram, axis=1)
            mfcc_mean[id] = np.mean(mfcc, axis=1) 
            mfcc_var[id] = np.var(mfcc, axis=1)
            mfcc_delta_mean[id] = np.mean(mfcc_delta, axis=1) 
            mfcc_delta_var[id] = np.var(mfcc_delta, axis=1)
            contrast_mean[id] = np.mean(contrast, axis=1) 
            contrast_var[id] = np.var(contrast, axis=1)
            poly_mean[id] = np.mean(poly_features, axis=1) 
            poly_var[id] = np.var(poly_features, axis=1)
            tonnetz_mean[id] = np.mean(tonnetz, axis=1) 
            tonnetz_var[id] = np.var(tonnetz, axis=1)
            harm_mean[id] = np.mean(harm_chroma_stft, axis=1) 
            harm_var[id] = np.var(harm_chroma_stft, axis=1)
            perc_mean[id] = np.mean(perc_chroma_stft, axis=1) 
            perc_var[id] = np.var(perc_chroma_stft, axis=1)
            id = id+1
        except:
            id = id+1
    feature_set['song_name'] = songname_vector
    feature_set['tempo'] = tempo_vector
    feature_set['duration'] = duration_vector
    feature_set['rms_mean'] = rms_mean
    feature_set['rms_var'] = rms_var
    feature_set['zcr_mean'] = zcr_mean
    feature_set['zcr_var'] = zcr_var
    feature_set['cent_mean'] = cent_mean
    feature_set['cent_var'] = cent_var
    feature_set['spec_bw_mean'] = spec_bw_mean
    feature_set['spec_bw_var'] = spec_bw_var
    feature_set['rolloff_mean'] = rolloff_mean
    feature_set['rolloff_var'] = rolloff_var
    feature_set['frame_mean'] = frame_mean
    feature_set['frame_var'] = frame_var
    chroma_stft_mean = pd.DataFrame(data=chroma_stft_mean, columns = ['chroma_stft_mean'+format(a,'02') for a in range(1, 1+chroma_stft_mean.shape[1])])
    chroma_stft_var = pd.DataFrame(data=chroma_stft_var, columns = ['chroma_stft_var'+format(a,'02') for a in range(1, 1+chroma_stft_var.shape[1])])
    chroma_cq_mean = pd.DataFrame(data=chroma_cq_mean, columns = ['chroma_cq_mean'+format(a,'02') for a in range(1, 1+chroma_cq_mean.shape[1])])
    chroma_cq_var = pd.DataFrame(data=chroma_cq_var, columns = ['chroma_cq_var'+format(a,'02') for a in range(1, 1+chroma_cq_var.shape[1])])
    chroma_cens_mean = pd.DataFrame(data=chroma_cens_mean, columns = ['chroma_cens_mean'+format(a,'02') for a in range(1, 1+chroma_cens_mean.shape[1])])
    chroma_cens_var = pd.DataFrame(data=chroma_cens_var, columns = ['chroma_cens_var'+format(a,'02') for a in range(1, 1+chroma_cens_var.shape[1])])
    mel_mean = pd.DataFrame(data=mel_mean, columns = ['mel_mean'+format(a,'03') for a in range(1, 1+mel_mean.shape[1])])
    mel_var = pd.DataFrame(data=mel_var, columns = ['mel_var'+format(a,'03') for a in range(1, 1+mel_var.shape[1])])
    mfcc_mean = pd.DataFrame(data=mfcc_mean, columns = ['mfcc_mean'+format(a,'02') for a in range(1, 1+mfcc_mean.shape[1])])
    mfcc_var = pd.DataFrame(data=mfcc_var, columns = ['mfcc_var'+format(a,'02') for a in range(1, 1+mfcc_var.shape[1])])
    mfcc_delta_mean = pd.DataFrame(data=mfcc_delta_mean, columns = ['mfcc_delta_mean'+format(a,'02') for a in range(1, 1+mfcc_delta_mean.shape[1])])
    mfcc_delta_var = pd.DataFrame(data=mfcc_delta_var, columns = ['mfcc_delta_var'+format(a,'02') for a in range(1, 1+mfcc_delta_var.shape[1])])
    contrast_mean = pd.DataFrame(data=contrast_mean, columns = ['contrast_mean'+format(a,'02') for a in range(1, 1+contrast_mean.shape[1])])
    contrast_var = pd.DataFrame(data=contrast_var, columns = ['contrast_var'+format(a,'02') for a in range(1, 1+contrast_var.shape[1])])
    poly_mean = pd.DataFrame(data=poly_mean, columns = ['poly_mean'+format(a,'02') for a in range(1, 1+poly_mean.shape[1])])
    poly_var = pd.DataFrame(data=poly_var, columns = ['poly_var'+format(a,'02') for a in range(1, 1+poly_var.shape[1])])
    tonnetz_mean = pd.DataFrame(data=tonnetz_mean, columns = ['tonnetz_mean'+format(a,'02') for a in range(1, 1+tonnetz_mean.shape[1])])
    tonnetz_var = pd.DataFrame(data=tonnetz_var, columns = ['tonnetz_var'+format(a,'02') for a in range(1, 1+tonnetz_var.shape[1])])
    harm_mean = pd.DataFrame(data=harm_mean, columns = ['harm_mean'+format(a,'02') for a in range(1, 1+harm_mean.shape[1])])
    harm_var = pd.DataFrame(data=harm_var, columns = ['harm_var'+format(a,'02') for a in range(1, 1+harm_var.shape[1])])
    perc_mean = pd.DataFrame(data=perc_mean, columns = ['perc_mean'+format(a,'02') for a in range(1, 1+perc_mean.shape[1])])
    perc_var = pd.DataFrame(data=perc_var, columns = ['perc_var'+format(a,'02') for a in range(1, 1+perc_var.shape[1])])
    feature_set = pd.concat([feature_set,chroma_stft_mean,chroma_stft_var,chroma_cq_mean,chroma_cq_var,chroma_cens_mean,chroma_cens_var,mel_mean,mel_var,
                mfcc_mean,mfcc_var,mfcc_delta_mean,mfcc_delta_var,contrast_mean,contrast_var,poly_mean,poly_var,tonnetz_mean,tonnetz_var,
                harm_mean,harm_var,perc_mean,perc_var], axis=1)
    feature_set.to_csv('Song_features.csv', mode='a', index=False, header=False)
    newpath="./static/"+file_name
    os.rename(songname, newpath)
    #if os.path.isfile(songname):
        #os.remove(songname)
    return file_name