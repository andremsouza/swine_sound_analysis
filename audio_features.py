#!/usr/bin/env python
# coding: utf-8
# %% [markdown]

# # Extração de *Features* de dados de áudio
# Nesse documento, se encontram as *features* extraídas de um subconjunto dos
# dados de áudio do projeto de TCC.
#
# A maioria das *features* selecionadas é proveniente do spectrograma de cada
# exemplo do conjunto de dados utilizado.
#
# Devido à natureza dos dados, acredito que seja possível extrair *features*
# relacionadas com séries temporais. Inicialmente, estou pesquisando os métodos
# implementados nesse [repositório](https://github.com/FelSiq/ts-pymfe).
#
# Por fim, utilizei as *features* selecionadas como entrada de um mapa de
# Kohonen, porém ainda não mapeei labels sobre o mapa resultante.
#
# Principais referências utilizadas:
#
# - https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.989&rep=rep1&type=pdf
# - https://librosa.org/doc/main/feature.html
# - https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b

# %% [markdown]
# # Imports

# %%
from datetime import datetime
# import IPython.display as ipd
import librosa
import librosa.display
# import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
# from pprint import pprint
import re
from scipy.signal import butter, filtfilt
# import seaborn as sns
# import swifter
import warnings

import sys
sys.path.append('./python-som/')

# import python_som

# %% [markdown]
# # Function definitions


# %%
def extract_data_from_filename(fname: str) -> list:
    """Extract datetime and other fields from filename from a video."""
    pattern: str = (r"^.*ALA_(\w)" + r"\)?_(\d)" +
                    r"_(\d{4})-(\d{2})-(\d{2})" +
                    r"_(\d{2})-(\d{2})-(\d{2}).*$")
    match: re.Match = re.fullmatch(pattern, fname)  # type: ignore
    data = [
        datetime(int(match.groups()[2]), int(match.groups()[3]),
                 int(match.groups()[4]), int(match.groups()[5]),
                 int(match.groups()[6]), int(match.groups()[7])), fname,
        match.groups()[0],
        int(match.groups()[1])
    ]
    return data


def load_audio_data(fname: str) -> list:
    data = extract_data_from_filename(fname)
    x, sr = librosa.load(fname)
    return data + [x, sr]


def butter_highpass(data: np.array, cutoff: float, fs: float, order: int = 5):
    """
    Design a highpass filter, removing noise from frequencies lower than
    cutoff.
    Args:
    - cutoff (float) : the cutoff frequency of the filter.
    - fs     (float) : the sampling rate.
    - order    (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs
    # design filter
    high = cutoff / nyq
    b, a = butter(order, high, btype='high', analog=False)
    # returns the filter coefficients: numerator and denominator
    y = filtfilt(b, a, data)
    return y


def timed_onset_samples(onset_samples: np.array, sr: int,
                        min_time: float) -> np.array:
    """Process onset samples to have a min_time inbetween, for segmentation"""
    processed_samples: list = [onset_samples[0]]  # First sample for comparison
    for i in range(1, onset_samples.shape[0]):
        if onset_samples[i] > processed_samples[-1] + sr * min_time:
            processed_samples.append(onset_samples[i])
    return np.array(processed_samples)


def extract_feature_means(audio_file_path: str,
                          verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("File:", audio_file_path)
    # config settings
    number_of_mfcc = 20

    # 0. Extracting info from filename
    if verbose:
        print("0.Extracting info from filename...")
    datetime, _, ala, grupo = extract_data_from_filename(audio_file_path)

    # 1. Importing 1 file
    if verbose:
        print("1.Importing file with librosa...")
    try:
        y, sr = librosa.load(audio_file_path)
    except Exception as e:
        print(e)
        return None

    # Trim leading and trailing silence from an audio signal
    # (silence before and after the actual audio)
    signal, _ = librosa.effects.trim(y)

    # 2. Fourier Transform
    if verbose:
        print("2.Fourier transform...")
    # Default FFT window size
    n_fft = 2048  # FFT window size
    hop_length = 512  # number audio of frames between STFT columns
    # (looks like a good default)

    # Short-time Fourier transform (STFT)
    y, sr = librosa.load(audio_file_path)
    d_audio = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

    # 3. Spectrogram
    if verbose:
        print("3.Spectrogram...")
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    # 4. Create the Mel Spectrograms
    if verbose:
        print("4.Mel spectograms...")
    s_audio = librosa.feature.melspectrogram(signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # 5 Zero crossings
    if verbose:
        print("5.Zero crossings...")

    # #6. Harmonics and Perceptrual
    if verbose:
        print("6.Harmonics and perceptrual...")
    # Note:
    #
    # Harmonics are characteristichs that represent the sound color
    # Perceptrual shock wave represents the sound rhythm and emotion
    y_harm, y_perc = librosa.effects.hpss(signal)

    # 7. Spectral Centroid
    if verbose:
        print("7.Spectral centroid...")
    # Note: Indicates where the ”centre of mass” for a sound is located and is
    # calculated as the weighted mean of the frequencies present in the sound.

    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids,
                                                          order=2)

    # spectral_centroid_feats = np.stack((spectral_centroids,
    # delta, accelerate))  # (3, 64, xx)

    # 8. Chroma Frequencies¶
    if verbose:
        print("8.Chroma frequencies...")
    # Note: Chroma features are an interesting and powerful representation
    # for music audio in which the entire spectrum is projected onto 12 bins
    # representing the 12 distinct semitones ( or chromas) of the musical
    # octave.

    # Increase or decrease hop_length to change how granular you want your data
    # to be
    hop_length = 512

    # Chromogram
    chromagram = librosa.feature.chroma_stft(signal,
                                             sr=sr,
                                             hop_length=hop_length)

    # 9. Tempo BPM (beats per minute)¶
    if verbose:
        print("9.Tempo BPM...")
    # Note: Dynamic programming beat tracker.

    # Create Tempo BPM variable
    tempo_y, _ = librosa.beat.beat_track(signal, sr=sr)

    # 10. Spectral Rolloff
    if verbose:
        print("10.Spectral rolloff...")
    # Note: Is a measure of the shape of the signal.
    # It represents the frequency below which a specified
    #  percentage of the total spectral energy(e.g. 85 %) lies.

    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr)[0]

    # spectral flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # Spectral Bandwidth¶
    # The spectral bandwidth is defined as the width of the band of light
    # at one-half the peak
    # maximum (or full width at half maximum [FWHM]) and is represented
    # by the two vertical
    # red lines and λSB on the wavelength axis.
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal,
                                                              sr=sr,
                                                              p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal,
                                                              sr=sr,
                                                              p=4)[0]

    audio_features = {
        "datetime":
        datetime,
        "ala":
        ala,
        "grupo":
        grupo,
        "file_name":
        audio_file_path,
        "zero_crossing_rate":
        np.mean(librosa.feature.zero_crossing_rate(signal)[0]),
        "zero_crossings":
        np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram":
        np.mean(db_audio[0]),
        "mel_spectrogram":
        np.mean(s_db_audio[0]),
        "harmonics":
        np.mean(y_harm),
        "perceptual_shock_wave":
        np.mean(y_perc),
        "spectral_centroids":
        np.mean(spectral_centroids),
        "spectral_centroids_delta":
        np.mean(spectral_centroids_delta),
        "spectral_centroids_accelerate":
        np.mean(spectral_centroids_accelerate),
        "chroma1":
        np.mean(chromagram[0]),
        "chroma2":
        np.mean(chromagram[1]),
        "chroma3":
        np.mean(chromagram[2]),
        "chroma4":
        np.mean(chromagram[3]),
        "chroma5":
        np.mean(chromagram[4]),
        "chroma6":
        np.mean(chromagram[5]),
        "chroma7":
        np.mean(chromagram[6]),
        "chroma8":
        np.mean(chromagram[7]),
        "chroma9":
        np.mean(chromagram[8]),
        "chroma10":
        np.mean(chromagram[9]),
        "chroma11":
        np.mean(chromagram[10]),
        "chroma12":
        np.mean(chromagram[11]),
        "tempo_bpm":
        tempo_y,
        "spectral_rolloff":
        np.mean(spectral_rolloff),
        "spectral_flux":
        np.mean(onset_env),
        "spectral_bandwidth_2":
        np.mean(spectral_bandwidth_2),
        "spectral_bandwidth_3":
        np.mean(spectral_bandwidth_3),
        "spectral_bandwidth_4":
        np.mean(spectral_bandwidth_4),
    }

    # extract mfcc feature
    mfcc_df = extract_mfcc_feature_means(audio_file_path,
                                         signal,
                                         sample_rate=sr,
                                         number_of_mfcc=number_of_mfcc)

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on='file_name')

    if verbose:
        print("DONE:", audio_file_path)
    return df

    # librosa.feature.mfcc(signal)[0, 0]


def extract_mfcc_feature_means(audio_file_name: str, signal: np.ndarray,
                               sample_rate: int,
                               number_of_mfcc: int) -> pd.DataFrame:

    mfcc_alt = librosa.feature.mfcc(y=signal,
                                    sr=sample_rate,
                                    n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {
        "file_name": audio_file_name,
    }

    for i in range(0, number_of_mfcc):
        # dict.update({'key3': 'geeks'})

        # mfcc coefficient
        key_name = "".join(['mfcc', str(i)])
        mfcc_value = np.mean(mfcc_alt[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc delta coefficient
        key_name = "".join(['mfcc_delta_', str(i)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc accelerate coefficient
        key_name = "".join(['mfcc_accelerate_', str(i)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df


# %% [markdown]
# # Extracting raw audio from .mp4 files
# Using librosa

# %%
DATA_DIR: str = '/home/rodrigo/1_SWINE_PROJECT/'
EXTENSION: str = '.mp4'
fnames: list = sorted([
    os.path.join(root, file) for root, _, files in os.walk(DATA_DIR)
    for file in files if file.endswith(EXTENSION)
])
pattern: str = (r"^.*ALA_(\w)" + r"\)?_(\d)" + r"_(\d{4})-(\d{2})-(\d{2})" +
                r"_(\d{2})-(\d{2})-(\d{2}).*$")
matches: list = [re.fullmatch(pattern, fname) for fname in fnames]
if len(matches) != len(fnames):
    raise ValueError("check fname patterns")
rows = np.array([[
    datetime(int(matches[i].groups()[2]), int(matches[i].groups()[3]),
             int(matches[i].groups()[4]), int(matches[i].groups()[5]),
             int(matches[i].groups()[6]), int(matches[i].groups()[7])),
    fnames[i], matches[i].groups()[0],
    int(matches[i].groups()[1])
] for i in range(len(matches))])
columns = ['datetime', 'fname', 'ala', 'grupo']
print('Number of identified audio files:', len(fnames))

# %% [markdown]
# ## Creating DataFrame with filenames (audios dataset)
# ### Loading filenames into DataFrame

# %%
audios = pd.DataFrame(data=rows, columns=columns)
audios.set_index(['datetime'], drop=True, inplace=True, verify_integrity=False)

# %% [markdown]
# ### Filtering by time

# %%
audios = audios.between_time('9:00', '12:00')

# %% [markdown]
# ### Extracting features (parallel)

# %%
print('Processing', len(audios), 'audios...')
result = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with mp.Pool(processes=64) as pool:
        result = pool.map(extract_feature_means, list(audios['fname']))
print("Done processing audios. Concatenating and writing to output file...")
for idx, i in enumerate(result):
    if i is None:
        del result[idx]
audios_means = pd.concat(result)
output_path = './features_means.csv'
audios_means.to_csv(output_path, index=False)

# %% [markdown]
# ## Visualizing DataFrame with features

# %%
