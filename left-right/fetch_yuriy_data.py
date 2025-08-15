from scipy.io import loadmat
import numpy as np


def get_yuriy_data(samples_around=100):
  X_1, Y_1, meta_data_1 = get_file('/Users/alejandro/VSCodeProjects/Models/yuriy-data/FREEFORMSubjectC1512102StLRHand.mat', samples_around)
  X_2, Y_2, _ = get_file('/Users/alejandro/VSCodeProjects/Models/yuriy-data/FREEFORMSubjectC1512082StLRHand.mat', samples_around)
  # X_3, Y_3, _ = get_file('/Users/alejandro/VSCodeProjects/Models/yuriy-data/FREEFORMSubjectB1511112StLRHand.mat', samples_around)

  X = np.concatenate([X_1, X_2])
  Y = np.concatenate([Y_1, Y_2])

  return X, Y, meta_data_1


def get_file(file_path, samples_around=100):
  mat_data = loadmat(file_path)
  data = mat_data['o'][0,0]

  meta_data = {
    'id': str(data['id'][0]),
    'tag': str(data['tag'][0]),
    'sampFreq': int(data['sampFreq'][0][0]),
    'num_samples': int(data['nS'][0][0]),
    'channel_names': [str(channel[0][0]) for channel in data['chnames']],
    'binsuV': int(data['binsuV'][0][0]),
  }

  channel_names = meta_data['channel_names']

  # Find indices of A1 and A2 channels
  a1_idx = channel_names.index('A1')
  a2_idx = channel_names.index('A2')
  channels_to_remove = [a1_idx, a2_idx, -1]

  # Remove specified channels
  channel_names = np.delete(np.array(channel_names), channels_to_remove)
  eeg = np.delete(data['data'], channels_to_remove, axis=1)
  markers = data['marker'].T[0]

  meta_data['channel_names'] = channel_names

  X = []
  Y = []

  # Iterate through markers array
  for i in range(samples_around, len(markers) - samples_around):
      # Check if marker changes from 0 to 1 or 2
      if markers[i-1] == 0 and (markers[i] == 1 or markers[i] == 2):
          # Get 100 samples before and after the marker change
          segment = eeg[i-samples_around:i+samples_around]
          X.append(segment)
          Y.append(markers[i])

  # Convert lists to numpy arrays
  X = np.array(X)
  Y = np.array(Y)

  return X, Y, meta_data