# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from pprint import pprint
from scipy.signal import butter, filtfilt
# import subprocess

# %% [markdown]
# # Extracting raw audio from .mp4 files

# %%
DATA_DIR: str = './data/'
video_filenames: list = [
    fname for fname in os.listdir(DATA_DIR) if fname.endswith('.mp4')
]
print('Directory:', DATA_DIR)
pprint(video_filenames)

# %%
# ? Removing audio conversion (librosa can load from .mp4)
# for fname in video_filenames:
#     audio_fname = fname[:-4] + '.wav'
#     print(fname + " -> " + audio_fname)
#     p = subprocess.run([
#         '/home/andre/anaconda3/envs/audio/bin/ffmpeg', '-y', '-hide_banner',
#         '-loglevel', 'warning', '-i', DATA_DIR + fname, '-vn',
#         DATA_DIR + audio_fname
#     ],
#                        capture_output=True)
#     print('Output: ', end='')
#     pprint(p.stdout)
#     print('Err: ', end='')
#     pprint(p.stderr)

# %% [markdown]
# # Selecting a random audio from subset and experimenting with it

# %%
audio_filenames: list = [
    fname for fname in os.listdir(DATA_DIR) if fname.endswith('.mp4')
]
print('Directory:', DATA_DIR)
pprint(audio_filenames)

# %%
audio_filename: str = DATA_DIR + np.random.choice(audio_filenames)
print("Loading: ", audio_filename)
x, sr = librosa.load(audio_filename)
print(type(x), type(sr))
# TODO: Verify if sampling rate (sr) is correct

# %%
# Display waveform
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.draw()

# %%
# Display spectogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(
    Xdb,
    sr=sr,
    x_axis='time',
    y_axis='hz',
)
plt.draw()

# %%
# Zero Crossings
zero_crossing = librosa.zero_crossings(x)
pprint(zero_crossing)
print("Zero crossing rate:", sum(zero_crossing) / len(zero_crossing))

# %%
# High-pass filter (?)

# %%
# Onset detection
o_env = librosa.onset.onset_strength(x, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
onset_frames = librosa.util.peak_pick(o_env, 3, 3, 3, 5, 0.3, 100)
onset_detection = librosa.onset.onset_detect(x, sr=sr)

# %%
D = np.abs(librosa.stft(x))

plt.figure(figsize=(15, 10))

ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         x_axis='time',
                         y_axis='log')
plt.title('Power spectrogram')

plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames],
           0,
           o_env.max(),
           color='r',
           alpha=0.9,
           linestyle='--',
           label='Onsets')

plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.draw()

# %%
