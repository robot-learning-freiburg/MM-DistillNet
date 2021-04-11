# -*- coding: utf-8 -*-
"""
Due to space constraints, we provide the audio in mp3 instead of wav
Librosa is known to be slow with mp3 handling, so before training,
we convert the mp3 to a pkl with the required audio format
"""
import argparse
import glob
import pickle
import os

import librosa

import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get the audio inputs')
parser.add_argument('--dir', help='The drive directory or the dataset path')
args = parser.parse_args()

# The script can be run for a single drive or for the whole
# dataset. We use the 'drive' string to decide which of these
# cases we are dealing with
mp3_files = glob.glob(os.path.join(
    args.dir,
    'audio/*mp3' if 'drive' in args.dir else '*/audio/*mp3',
))

for audio in tqdm(mp3_files):
    y, sr = librosa.load(audio, sr=44100)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    with open(audio.replace('.mp3', '.pkl'), 'wb') as handle:
        pickle.dump(S_dB, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(audio.replace('.mp3', '.pkl'))
