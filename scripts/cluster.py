import re
import traceback
import librosa
import sklearn
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
data = pd.read_csv('Song_features.csv') 

def rec(file_name):
    data = pd.read_csv('Song_features.csv')
    data = data.dropna() 
    x = data.loc[:, 'tempo':].values
    stdScaler = StandardScaler()
    stdScaler.fit(x)
    #std_data = pd.DataFrame(stdScaler.transform(x), columns=data.loc[:,'tempo':].columns)
    #Nstd_PC = 6
    #std_pca = PCA(n_components=Nstd_PC)
    #std_PCs = std_pca.fit_transform(std_data.values)
    #std_PCs = pd.DataFrame(data = std_PCs) 
    kmeans = KMeans(n_clusters=30, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km = kmeans.fit(x)
    cluster_map = pd.DataFrame()
    cluster_map['song_name'] = data['song_name']
    cluster_map['cluster'] = km.labels_
    # Get cluster of the input song
    input_cluster = cluster_map[cluster_map.song_name == file_name].cluster.values[0]

    # Get only the songs in the same cluster, excluding the input song
    same_cluster_song_names = cluster_map[(cluster_map.cluster == input_cluster) & (cluster_map.song_name != file_name)]['song_name']

    rec = data[data.song_name.isin(same_cluster_song_names)]  # only same-cluster songs
    distance = []
    song = data[(data.song_name == file_name)].head(1).values[0]
    rec = data[data.song_name != file_name]
    for songs in rec.values:
        d = 0
        for col in np.arange(len(rec.columns)):
            if not col in [0]:
                d = d + np.absolute(float(song[col]) - float(songs[col]))
        distance.append(d)
    rec['distance'] = distance
    rec = rec.sort_values('distance')
    columns = ['song_name']
    recs=rec[columns][:10]
    listtt=recs.values.tolist()
    return listtt