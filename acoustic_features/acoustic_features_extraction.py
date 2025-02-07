import librosa
import pandas as pd
import numpy as np
import glob
import os

def get_loudness_features(y, sr):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    return np.mean(rms), np.std(rms), (np.max(rms)-np.min(rms))

def get_pitch_features(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_values = f0[np.logical_not(np.isnan(f0))]
    if f0_values.shape[0] > 0:
        return np.mean(f0_values), np.std(f0_values), (np.max(f0_values) - np.min(f0_values))
    else:
        return 0., 0., 0.



path = '/raid/magics/corpora/switchboard/utterances/' #utterances path on the Chupacabra server to do analysis on the Switchboard dataset
list_of_files = []
feature_vector = []

for root, dirs, files in os.walk(path):
    for file_ in files:
        list_of_files.append(os.path.join(root, file_))

for file_ in list_of_files:
    print('Processing', file_)
    y, sr = librosa.load(file_)
    file_vector = []
    file_vector.append(file_)
    file_vector.extend(get_pitch_features(y, sr))
    file_vector.extend(get_loudness_features(y, sr))
    feature_vector.append(file_vector)

print('Creating DataFrame.')
df = pd.DataFrame(feature_vector, columns=['file name',
                                           'pitch mean', 'pitch std dev', 'pitch range',
                                           'loudness mean', 'loudness std dev', 'loudness range'])
df.to_csv('acoustic_features.csv', index=False)
