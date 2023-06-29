#File is used in the display of EEG data from a .npy file. Can be done based on previously knowing the duration of the signal collection process
import numpy as np
import matplotlib.pyplot as plt

data = np.load(
    "C:/Users/prith/Desktop/final_uni_sem/thesis_B/test_data_draft/15_sec_data.npy")


div_by_5 = data.shape[1]/5
sampling_rate = 128

# Calculate the duration in seconds


duration = div_by_5 / sampling_rate
print(duration)
# Create time axis for the plot
time = np.linspace(0, duration, len(data))
print(time)
# Plot the time-domain graph
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Graph')
plt.grid(True)
plt.show()
