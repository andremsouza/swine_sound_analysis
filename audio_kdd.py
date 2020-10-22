# %% [markdown]
# # Exporatory analysis of features extracted from swine audios
# This dataset contains the means of the extracted features, for each audio
# Many descriptions of features were taken from the following books:
# - https://books.google.com/books?hl=en&lr=&id=AF30yR41GIAC&oi=fnd&pg=PP9&dq=Signal+Processing+Methods+for+Music+Transcription&ots=Ooq77iPfKD&sig=WFyZTyVVFtagCBGCRLWpV8rY-Tk

# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display as lrdisp
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# # Loading dataset

# %%
df = pd.read_csv('features_means_0900_1200.csv', index_col=0, verbose=True)
df.index = pd.to_datetime(df.index)
df['rac'] = False
df.loc['2020-09-22':, 'rac'] = True  # type: ignore

# %% [markdown]
# ## Visualizing waveform and spectrogramof random sample

# %%
file_name = np.random.choice(df['file_name'])
x, sr = lr.load(file_name)

fig, ax = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(16, 9))
lrdisp.waveplot(
    x,
    sr=sr,
    ax=ax[0],
)
img = lrdisp.specshow(lr.amplitude_to_db(np.abs(lr.stft(x)), ref=np.max),
                      y_axis='log',
                      x_axis='time',
                      ax=ax[1])
fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
plt.draw()
fig.savefig('./audio_sample.png', bbox_inches='tight')

# %% [markdown]
# # Visualizing basic dataset properties

# %% [markdown]
# ### Shape

# %%
print("Shape:", df.shape)
print(df.loc[:, 'rac'].groupby(df.loc[:, 'rac']).count())
print(df.info())

# %% [markdown]
# Due to the high amount of extracted features, there must be a selection
# of the most relevant ones for further analysis with advanced methods
# (e.g., SOM)

# %% [markdown]
# ### Class distribution

# %%
df_melt = pd.melt(df, value_vars=['rac'], value_name='ractopamine')
plt.figure(figsize=(10, 10))
sns.set(style="whitegrid",
        palette=sns.color_palette("muted", n_colors=6, desat=1.0))
ax = sns.countplot(data=df_melt, x='ractopamine', hue='ractopamine')

for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.2, p.get_height()),
                ha='center',
                va='top',
                color='white',
                size=18)

plt.draw()

# %% [markdown]
# There's a substantial imbalance between the subsets.
# It's not an obstacle for this analysis. However, for the next steps,
# it may be desirable to perform an upsampling/downsampling on the dataset.

# %% [markdown]
# ### Zero Crossings
# The Zero crossing rate (ZCR) measures the number of times that the time
# domain signal changes its sign. Even though it is computed in the time
# domain, it describes the amount of high-frequency energy in the signal
# (i.e., 'brightness') and correlates strongly with the spectral centroid
# ZCR has also proven to be quite discriminative for classes of percussion
# instruments.

# %%
print(df.loc[:, ['zero_crossing_rate', 'zero_crossings']].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=['zero_crossing_rate'])
sns.set(style="whitegrid",
        palette=sns.color_palette("muted", n_colors=6, desat=1.0))
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    kde=True,
    rug=True,
    row='variable',
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# The distribution of the zero crossing seems slightly different after the
# ractopamine provision.
#
# As noted by Ricardo, this may be related to an increase in the animals'
# vocal activity.

# %% [markdown]
# ### Spectrogram and Mel-Spectrogram
# There exists only one Fourier transform of a given signal.
# However, there is an infinite number of time-frequency representations.
# The most popular one is the spectrogram, defined as the Fourier transform of
# successive signal frames. Frames are widely used in audio processing
# algorithms. They are portions of the signal with given time localizations.
#
# Standard window shapes are Gaussian, Hamming, Hanning, or rectangular and
# typical frame durations are from 20 ms to 100 ms in audio processing. As a
# rule of thumb, rectangular windows should not be used in practice, except
# under special circumstances. From frames, it is easy to build short time
# Fourier transforms (STFTs) as the FT of successive frames.
#
# Spectrograms are energy representations and they are defined as the squared
# modulus of the STFT.
#
# The mel-spectrogram is an spectrogram, converted to the Mel scale

# %%
print(df.loc[:, ['spectrogram', 'mel_spectrogram']].describe())
df_melt = pd.melt(df,
                  id_vars=['rac'],
                  value_vars=['spectrogram', 'mel_spectrogram'])
sns.set(style="whitegrid",
        palette=sns.color_palette("muted", n_colors=6, desat=1.0))
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# No relevant differences in the distributions were observed.
#
# These features are discard candidates.
#
# Further analysis required with correlation heatmap.

# %% [markdown]
# ### Harmonics and Perceptual Shock Wave
# Harmonics are characteristichs that represent the sound color
# Perceptrual shock wave represents the sound rhythm and emotion

# %%
print(df.loc[:, ['harmonics', 'perceptual_shock_wave']].describe())
df_melt = pd.melt(df,
                  id_vars=['rac'],
                  value_vars=['harmonics', 'perceptual_shock_wave'])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# No relevant differences in the distributions were observed.
#
# These features are discard candidates.
#
# Further analysis required with correlation heatmap.

# %% [markdown]
# ### Spectral Centroids
# In addition to the rough spectrum described by the MFCCs and bandwise
# energy descriptors, more simple spectral shape features are also useful.
# These include the first four moments of the spectrum, i.e.,
# spectral centroid, spectralspread, spectral skewness, and spectral kurtosis.
#
# The spectral centroid is the "center-of-mass" of the analysed audio file.

# %%
print(df.loc[:, ['spectral_centroids']].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=['spectral_centroids'])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# The distribution of the spectral centroids seems to have a displacement to
# higher values, after the provision of ractopamine to the animals.
#
# This seems to suggest that the most intense peaks of audio were at higher
# frequencies, with rac=True.

# %% [markdown]
# ### Spectral Centroids (Delta and Accelerate)

# %%

print(df.loc[:, ['spectral_centroids_delta', 'spectral_centroids_accelerate']].
      describe())
df_melt = pd.melt(
    df,
    id_vars=['rac'],
    value_vars=['spectral_centroids_delta', 'spectral_centroids_accelerate'])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# No relevant differences in the distributions were observed.
#
# These features are discard candidates.
#
# Further analysis required with correlation heatmap.

# %% [markdown]
# ### Chroma features
# The chroma vector is a perceptually motivated feature vector using the
# concept of chroma in Shepard's helix representation of musical pitch
# perception. According to Shepard [584], the perception of pitch with respect
# to a musical context can be graphically represented by using a continually
# cyclic helix that has two dimensions, chroma and height. Chroma refers to the
# position of a musical pitch within an octave that corresponds to a cycle of
# the helix.

# %%
chroma_cols = [
    'chroma1',
    'chroma2',
    'chroma3',
    'chroma4',
    'chroma5',
    'chroma6',
    'chroma7',
    'chroma8',
    'chroma9',
    'chroma10',
    'chroma11',
    'chroma12',
]
print(df.loc[:, chroma_cols].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=chroma_cols)
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    row='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# Chroma features may be very useful for distinguishing musical tones.
# However, related research papers didn't seem to use it for pig audio
# analysis. With this, they are discard candidates.
#
# Still, there were some interesting changes in distribution for chroma7,
# chroma8, chroma9, chroma11, and chroma12, which should be further analyzed.

# %%
plt.figure(figsize=(20, 20))
sns.heatmap(df.loc[:, chroma_cols + ['rac']].corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            center=0.0,
            robust=True,
            annot=True)
plt.draw()

# %% [markdown]
# As expected, the chroma features are highly correlated with their neighbors.
#
# When comparing to our class (rac), the hightest absolute correlation between
# it and a chroma feature is 0.15 which is not very substantial.

# %% [markdown]
# ### Tempo (BMP)
# Estimate mean of the tempo (beats per minute), as extracted by librosa.

# %%
print(df.loc[:, ['tempo_bpm']].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=['tempo_bpm'])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# This feature, similarly to the chroma features, is mostly used with music
# analysis. No relevant differences in the distributions were observed.
#
# This feature is a discard candidate.
#
# Further analysis required with correlation heatmap.

# %% [markdown]
# ### Spectral Rolloff (85%)
# Spectral roll-off is defined as the frequency index R below which a certain
# fraction (85%) of the spectral energy resides.

# %%
print(df.loc[:, ['spectral_rolloff']].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=['spectral_rolloff'])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# A slight displacement can be seem between the two groups of files.
# This could reinforce the hypothesis drawn from the spectral centroid
# analysis, as the 85% threshold is at a higher frequency after RAC provision.

# %% [markdown]
# ### Spectral Flux
# The spectral flux, also known as the delta spectrum magnitude, is a measure
# of local spectral change. It is defined as the squared norm of the
# frame-to-frame spectral difference.

# %%
print(df.loc[:, ['spectral_flux']].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=['spectral_flux'])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# There has been no noticeable major changes in the distribution.
# This suggests that the number of "peaks"/onsets is similar after the rac
# provision.
# This feature is a discard candidate.

# %% [markdown]
# ### Spectral Bandwidth
# The bandwidth of the spectrum is described by spectral spread
#
# The spectral bandwidth 1 at frame t is computed by:
#
# (sum_k S[k, t] * (freq[k, t] - centroid[t])**p)**(1/p)

# %%
print(df.loc[:, [
    'spectral_bandwidth_2', 'spectral_bandwidth_3', 'spectral_bandwidth_4'
]].describe())
df_melt = pd.melt(df,
                  id_vars=['rac'],
                  value_vars=[
                      'spectral_bandwidth_2', 'spectral_bandwidth_3',
                      'spectral_bandwidth_4'
                  ])
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    col='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# No relevant differences in the distributions were observed.
#
# These features are discard candidates.
#
# Further analysis required with correlation heatmap.

# %% [markdown]
# ### Correlation matrix of subset of features

# %%
plt.figure(figsize=(20, 20))
sns.heatmap(df.loc[:, [
    'zero_crossing_rate', 'spectrogram', 'mel_spectrogram', 'harmonics',
    'perceptual_shock_wave', 'spectral_centroids', 'tempo_bpm',
    'spectral_rolloff', 'spectral_flux', 'spectral_bandwidth_2',
    'spectral_bandwidth_3', 'spectral_bandwidth_4', 'rac'
]].corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            center=0.0,
            robust=True,
            annot=True)
plt.draw()

# %% [markdown]
# The spectral centroids feature had the highest correlation with 'rac',
# between the analyzed features.

# %% [markdown]
# ### MFCC
# Mel-frequency cepstral coefficients (MFCCs) describe the rough shape of
# the signal spectrum and are widely used in speech recognition [536].
# Similarly, they are often encountered in percussion transcription algorithms.
# Usually the coefficients are calculated in short (about 20 ms) partially
# overlapping frames over the analysed segment, and 5 to 15 coefficients are
# retained in each frame. Instead of using these directly as features,
# typically the mean and variance of each coefficient over the segment are
# used. In addition, the first- and second-order temporal differences of the
# coefficients, and the means and variances of these, are commonly used as
# features.

# %%
mfcc_cols = [
    'mfcc0',
    'mfcc1',
    'mfcc2',
    'mfcc3',
    'mfcc4',
    'mfcc5',
    'mfcc6',
    'mfcc7',
    'mfcc8',
    'mfcc9',
    'mfcc10',
    'mfcc11',
    'mfcc12',
    'mfcc13',
    'mfcc14',
    'mfcc15',
    'mfcc16',
    'mfcc17',
    'mfcc18',
    'mfcc19',
]
print(df.loc[:, mfcc_cols].describe())
df_melt = pd.melt(df, id_vars=['rac'], value_vars=mfcc_cols)
sns.displot(
    data=df_melt,
    x='value',
    hue='rac',
    row='variable',
    kde=True,
    rug=True,
    height=9,
    aspect=1,
)
plt.draw()

# %% [markdown]
# From related papers, the MFCCs are some of the most representative features.
# The main question related the MFCCs is the number of coefficients that will
# be used. For most use cases, 12 or 13 coefficients seem to be enough.
#
# TODO Study and write detailed description of MFCC in monograph.
#
# mfcc14 through mfcc19 are discard candidates.

# %%
plt.figure(figsize=(20, 20))
sns.heatmap(df.loc[:, mfcc_cols + ['rac']].corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            center=0.0,
            robust=True,
            annot=True)
plt.draw()

# %% [markdown]
# Initially the highest absolute correlation with 'rac' is onm mfcc9, mfcc15,
# and mfcc16.
#
# The coefficients by themselfs seem to be relatively independent of each
# other, taking into account only the correlation matrix.
# This may be an indicator of high representativity of the dataset.

# %% [markdown]
# ### Visualizing discard candidates

# %%
discard_features = [
    'zero_crossings',
    'spectrogram',
    'mel_spectrogram',
    'harmonics',
    'perceptual_shock_wave',
    'tempo_bpm',
    'spectral_flux',
    'spectral_bandwidth_2',
    'spectral_bandwidth_3',
    'spectral_bandwidth_4',
] + chroma_cols + [
    'mfcc14',
    'mfcc15',
    'mfcc16',
    'mfcc17',
    'mfcc18',
    'mfcc19',
    'mfcc14_delta',
    'mfcc15_delta',
    'mfcc16_delta',
    'mfcc17_delta',
    'mfcc18_delta',
    'mfcc19_delta',
    'mfcc14_accelerate',
    'mfcc15_accelerate',
    'mfcc16_accelerate',
    'mfcc17_accelerate',
    'mfcc18_accelerate',
    'mfcc19_accelerate',
]

print('# of discarded features:', len(discard_features))
print(
    '# of remaining features:',
    df.loc[:, 'zero_crossing_rate':].shape[1] -  # type: ignore
    len(discard_features))

# %% [markdown]
# # Final comments
# After analyzing the extracted features for our dataset, the following
# features are discard candidates:
#
# - spectrogram';
# - mel-spectrogram';
# - harmonics';
# - perceptual shock wave;
# - chroma features;
# - tempo (bpm);
# - spectral flux;
# - spectral bandwidth;
# - mfcc14, ..., mfcc19.
#
# For now, the remaining features shall be used for further analysis and
# data visualization methods, such as SOMs.
#
# If necessary, more features, such as deltas and accelerates, can be
# discarded, for performance reasons. The relevance of these features should
# be further analyzed.

# %%

# %%
