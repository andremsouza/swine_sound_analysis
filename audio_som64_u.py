# %% [markdown]
# # Testing python-som with audio dataset

# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
# import librosa as lr
# import librosa.display as lrdisp
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sklearn.preprocessing

from python_som import SOM

FILE_PREFIX = 'som64_u'

# %% [markdown]
# # Loading dataset

# %%
df = pd.read_csv('features_means.csv', index_col=0, verbose=True)
df.index = pd.to_datetime(df.index)
df['rac'] = False
df.loc['2020-09-22':, 'rac'] = True  # type: ignore
df.sort_index(inplace=True)

# %% [markdown]
# ## Checking for and dropping duplicates

# %%
# Resetting index for duplicate analysis
df.reset_index(inplace=True)
print("Duplicates by filename:",
      df.duplicated(subset=['file_name']).value_counts(),
      sep='\n')
df.drop_duplicates(subset=['file_name'], inplace=True)
print("Duplicates by (datetime, ala, grupo):",
      df.duplicated(subset=['datetime', 'ala', 'grupo']).value_counts(),
      sep='\n')
df.drop_duplicates(subset=['datetime', 'ala', 'grupo'], inplace=True)
# Rebuilding dataframe index
df.set_index('datetime', inplace=True)

# %%
# Dropping tail of dataset for class balancing
# tail_size = abs(
#     len(df[df['rac'].astype(int) == 1]) - len(df[df['rac'].astype(int) == 0]))
# df.drop(df.tail(tail_size).index, inplace=True)

# %% [markdown]
# ## Visualizing distribution of sample dates

# %%
df_tmp = pd.DataFrame(df['file_name'].resample('1D').count())
df_tmp['count'] = df_tmp['file_name']
del df_tmp['file_name']
df_tmp['rac'] = False
df_tmp.loc['2020-09-22':, 'rac'] = True  # type: ignore

plt.figure(figsize=(10, 10))
sns.set(style="whitegrid",
        palette=sns.color_palette("muted", n_colors=6, desat=1.0))
sns.barplot(y=df_tmp.index, x=df_tmp['count'], hue=df_tmp['rac'])
plt.draw()

df_tmp = pd.DataFrame(df['file_name'].resample('1H').count())
df_tmp['count'] = df_tmp['file_name']
del df_tmp['file_name']
df_tmp['rac'] = False
df_tmp.loc['2020-09-22':, 'rac'] = True  # type: ignore
df_tmp = df_tmp.reset_index()
df_tmp['hour'] = df_tmp['datetime'].dt.hour

plt.figure(figsize=(10, 10))
sns.set(style="whitegrid",
        palette=sns.color_palette("muted", n_colors=6, desat=1.0))
sns.barplot(y=df_tmp['hour'], x=df_tmp['count'], hue=df_tmp['rac'], orient='h')
plt.draw()

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

# %%
# using sklearn's MinMaxScaler
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))

df_train = df.iloc[:, 3:-1].copy()
df_train = scaler.fit_transform(df_train)

# %%
# Defining first element of SOM shape
# Second element will be assigned based on the ratio between the
# first two principal components of the train dataset
som_x: int = 64
try:
    with open(f'./{FILE_PREFIX}.obj', 'rb') as f:
        som = pickle.load(f)
except FileNotFoundError:
    som = SOM(x=som_x,
              y=None,
              input_len=df_train.shape[1],
              learning_rate=0.5,
              neighborhood_radius=1.0,
              neighborhood_function='gaussian',
              cyclic_x=True,
              cyclic_y=True,
              data=df_train)
    # Training SOM
    som.weight_initialization(mode='linear', data=df_train)
    som.train(data=df_train, mode='random', verbose=True)
    with open(f'./{FILE_PREFIX}.obj', 'wb') as f:
        pickle.dump(som, f)

# %%
som_x, som_y = som.get_shape()
print('SOM shape:', (som_x, som_y))

# %%
# Visualizing distance matrix and activation matrix
umatrix = som.distance_matrix()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
sns.heatmap(umatrix.T, cmap='bone_r', ax=ax1, robust=True)
sns.heatmap(som.activation_matrix(data=df_train).T,
            cmap='mako',
            ax=ax2,
            robust=True)
ax1.invert_yaxis()
ax2.invert_yaxis()
fig.savefig(f'./output_{FILE_PREFIX}/{FILE_PREFIX}_umatrix_activation.png',
            bbox_inches='tight',
            transparent=True)
plt.draw()

# %% [markdown]
# ## Visualizing distribution of features

# %%
for column in df.iloc[:, 3:-1].columns:
    hmap = som.get_weights()[:, :, df.iloc[:, 3:-1].columns.get_loc(column)].T
    fig = plt.figure(figsize=(16, 9))
    ax = sns.heatmap(hmap, robust=True, cmap='BrBG')
    ax.invert_yaxis()
    fig.savefig(f'./output_{FILE_PREFIX}/{FILE_PREFIX}_{column}.png',
                bbox_inches='tight',
                transparent=True)
    plt.close(fig=fig)

# %% [markdown]
# ## Visualizing distribution of audios by metadata (day, hour, ...)
# Each node is colorized according to its most frequent label

# %%
df['days'] = df.index.date
df['days'] = (df['days'] - df['days'][0])
df['days'] = df['days'].apply(lambda x: x.days)
df['hour'] = df.index.hour

# %%
# Visualizing 'rac' distribution
class_assignments = som.label_map(np.array(df_train), np.array(df['rac']))
hmap = np.zeros((som_x, som_y))
for i, j in sorted(class_assignments.keys()):
    try:
        hmap[i][j] = class_assignments[(i, j)].most_common()[0][0] + 1
    except Exception:
        continue
hmap = hmap.T
fig = plt.figure(figsize=(16, 9))
ax = sns.heatmap(hmap,
                 cmap=sns.color_palette(palette=["#000000", "blue", "orange"],
                                        n_colors=3),
                 cbar_kws={'ticks': [0, 1, 2]})
ax.invert_yaxis()
fig.savefig(f'./output_{FILE_PREFIX}/{FILE_PREFIX}_rac.png',
            bbox_inches='tight',
            transparent=True)
plt.show()

# %%
# Visualizing by 'grupo'
print(df.groupby('grupo')['rac'].count())
column = 'grupo'
class_assignments = som.label_map(np.array(df_train), np.array(df[column]))
hmap = np.zeros((som_x, som_y))
for i, j in sorted(class_assignments.keys()):
    try:
        hmap[i][j] = class_assignments[(i, j)].most_common()[0][0]
    except Exception:
        hmap[i][j] = 0
hmap = hmap.T
fig = plt.figure(figsize=(16, 9))
ax = sns.heatmap(hmap,
                 cmap=sns.color_palette(palette=["#000000", "blue", "orange"],
                                        n_colors=3),
                 cbar_kws={'ticks': [0, 1, 2]})
ax.invert_yaxis()
fig.savefig(f'./output_{FILE_PREFIX}/{FILE_PREFIX}_{column}.png',
            bbox_inches='tight',
            transparent=True)
plt.show()

# %%
# Visualizing by 'days'
print(df.groupby('days')['rac'].count())
column = 'days'
class_assignments = som.label_map(np.array(df_train), np.array(df[column]))
hmap = np.zeros((som_x, som_y))
for i, j in sorted(class_assignments.keys()):
    try:
        hmap[i][j] = class_assignments[(i, j)].most_common()[0][0]
    except Exception:
        hmap[i][j] = -1
hmap = hmap.T
fig = plt.figure(figsize=(16, 9))
ax = sns.heatmap(hmap, cmap='viridis')
ax.invert_yaxis()
fig.savefig(f'./output_{FILE_PREFIX}/{FILE_PREFIX}_{column}.png',
            bbox_inches='tight',
            transparent=True)
plt.show()

# %%
# Visualizing by 'hour'
print(df.groupby('hour')['rac'].count())
column = 'hour'
class_assignments = som.label_map(np.array(df_train), np.array(df[column]))
hmap = np.zeros((som_x, som_y))
for i, j in sorted(class_assignments.keys()):
    try:
        hmap[i][j] = class_assignments[(i, j)].most_common()[0][0]
    except Exception:
        hmap[i][j] = -1
hmap = hmap.T
fig = plt.figure(figsize=(16, 9))
ax = sns.heatmap(hmap,
                 cmap=sns.diverging_palette(150,
                                            250,
                                            s=100,
                                            l=20,
                                            sep=1,
                                            n=26,
                                            center='light'),
                 center=12)
ax.invert_yaxis()
fig.savefig(f'./output_{FILE_PREFIX}/{FILE_PREFIX}_{column}.png',
            bbox_inches='tight',
            transparent=True)
plt.show()

# %%
