#This script handles the execution of the creation of the frequency domain graph and ERDS graph which allows for
#more in-depth information regarding the collection of data and its processing, allowing to individually check channels with more activity.
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def bandpass_filter(data, fs, low, high):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def plot_heatmap(ax, data, freqs, time_bins, channel, cmap='coolwarm'):
    img = ax.pcolormesh(time_bins, freqs, data, cmap=cmap, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Channel {channel + 1}')
    return img


def calculate_erd_ers(data, fs, freq_bins, nperseg):
    n_channels = data.shape[0]
    erd_ers = []

    for i, channel_data in enumerate(data):
        f, t, Zxx = signal.stft(channel_data, fs, nperseg=nperseg)
        power = np.abs(Zxx) ** 2
        valid_freqs = f[(f >= freq_bins[0]) & (f <= freq_bins[-1])]
        power = power[(f >= freq_bins[0]) & (f <= freq_bins[-1]), :]
        baseline = np.mean(power, axis=1, keepdims=True)
        erd_ers.append(10 * np.log10(power / baseline))

    return np.array(erd_ers)


def plot_fft(ax, fft_data, freqs, channel):
    ax.plot(freqs, fft_data)
    ax.set_xlim([0, high_freq])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Channel {channel + 1}')


def calculate_fft(data, fs):
    n_channels = data.shape[0]
    fft_data = []
    frequency_bands = {'delta': (0.5, 4),
                       'theta': (4, 8),
                       'alpha': (8, 13),
                       'beta': (13, 30),
                       'gamma': (30, 50)}
    useful_channels = {}
    band_averages = {}
    for channel in range(n_channels):
        for channel_data in data:
            channel_fft = np.abs(np.fft.rfft(channel_data))
            fft_result = np.fft.rfft(channel_data)
            freqs = np.fft.rfftfreq(channel_data.size, 1/fs)
            for f_band, (band_low, band_high) in frequency_bands.items():
                low = np.full_like(freqs, band_low)
                high = np.full_like(freqs, band_high)
                band_indices = np.where((freqs >= low) & (freqs <= high))
                band_averages[f_band] = np.mean(channel_fft[band_indices])
            fft_data.append(np.abs(fft_result))
            useful_channels[channel] = band_averages
    print(useful_channels)
    return np.array(fft_data)


# Load your EEG data
sample_data_folder = "C:/Users/prith/Desktop/final_uni_sem/thesis_B/test_data_draft"
sample_data_folder_join = os.path.join(sample_data_folder, '3.npy')
eeg_data = np.load(sample_data_folder_join)  # 加载EEG信号
#eeg_data = eeg_data[1:17, :]
print(eeg_data.shape)

fs = 250
low_freq = 4
high_freq = 40
freq_bins = np.linspace(low_freq, high_freq, 37)

# Bandpass filter the EEG data
filtered_data = np.array([bandpass_filter(
    channel_data, fs, low_freq, high_freq) for channel_data in eeg_data])

# Calculate ERD/ERS for each channel
nperseg = 128
erd_ers = calculate_erd_ers(filtered_data, fs, freq_bins, nperseg)

# Plot the ERD/ERS heatmap for all channels in a single figure
fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
axes = axes.flatten()

for i, channel_erd_ers in enumerate(erd_ers):
    f, t, Zxx = signal.stft(filtered_data[i], fs, nperseg=nperseg)
    valid_freqs = f[(f >= freq_bins[0]) & (f <= freq_bins[-1])]
    img = plot_heatmap(axes[i], channel_erd_ers, valid_freqs, t, i)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(img, cax=cbar_ax)
plt.show()

# Calculate FFT for each channel
fft_data = calculate_fft(filtered_data, fs)

# Plot the FFT results for all channels in a single figure
fig_fft, axes_fft = plt.subplots(
    4, 4, figsize=(16, 16), sharex=True, sharey=True)
axes_fft = axes_fft.flatten()

n_channels = fft_data.shape[0]  # Number of channels in fft_data
for i in range(n_channels):
    if i < len(axes_fft):  # Check if index is within the valid range
        plot_fft(axes_fft[i], fft_data[i], np.fft.rfftfreq(
            filtered_data[i].size, 1 / fs), i)

fig_fft.tight_layout()
plt.show()
