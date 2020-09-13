# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from pprint import pprint
import subprocess

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
for fname in video_filenames:
    audio_fname = fname[:-4] + '.wav'
    print(fname + " -> " + audio_fname)
    p = subprocess.run([
        '/home/andre/anaconda3/envs/audio/bin/ffmpeg', '-y', '-hide_banner',
        '-loglevel', 'warning', '-i', DATA_DIR + fname, '-vn',
        DATA_DIR + audio_fname
    ],
                       capture_output=True)
    print('Output: ', end='')
    pprint(p.stdout)
    print('Err: ', end='')
    pprint(p.stderr)

# %% [markdown]
# # Selecting a random audio from subset and experimenting with it

# %%
audio_filenames: list = [
    fname for fname in os.listdir(DATA_DIR) if fname.endswith('.wav')
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
zero_crossing = librosa.zero_crossings(x)

# %%
