import mne
import numpy as np
import matplotlib.pyplot as plt


path = "./300.npy"
raw_data = np.load(path)
data = raw_data[0:16, :]

print(data.shape)
print(data[2])


plt.plot(data[2])
plt.show()



info = mne.create_info(
    ch_names=['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8',
              'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15', 'EEG16'],
    ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
    sfreq=125
)


custom_raw = mne.io.RawArray(data, info)
print(custom_raw)


scaling = {'eeg': 2}
custom_raw.plot(n_channels=16,
                scalings=scaling,
                title='Data from arrays',
                show=True, block=True)

plt.show()
