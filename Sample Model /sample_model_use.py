#This script is used in testing the accuracy of different preprocessing methods with a CNN MLP 
#model with the training data used for our model. Note: This model was made for looking at particular files (.npy) 
#within our own Linux system. 

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from numpy import ndarray
from scipy import signal
from scipy.signal import cheby1, cheby2, filtfilt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



setup_seed(20)


sample_data_folder = "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1"
sample_data_test_folder = "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_label_draft_1"
file_indices = list(range(0, 16))
# file_indices_1 = list(range(0,4))
paths = ["/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 1",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 2",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 3",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 4",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 5",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 6",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 7",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 8",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 9",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 10",
         "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/Participant 11"]
label = 0
sampling_rate = 128
first_five = sampling_rate * 5
all_windows_with_labels = []
for path in paths:
    for i in file_indices:
        file_name = f"file_{i // 4}_{i % 4}.npy"
        file_path = os.path.join(path, file_name)
        num1, num2 = map(int, file_name.split("_")[1].split(".")[0]), int(file_name.split("_")[2].split(".")[0])
        print(file_name)
        # print(num2)
        if (num2 == 0):
            label = 0
        elif (num2 == 1):
            label = 1
        elif (num2 == 2):
            label = 2
        elif (num2 == 3):
            label = 3
        data = np.load(file_path)
        # print(data.shape)
        data = data[1:17, :]
        # print(data.shape)
        # data = data[:-first_five]
        # print(data.shape)
        lowcut = 1
        highcut = 45
        order = 4
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype="band")

        filtered_data = signal.lfilter(b, a, data, axis=1)
        # lowcut = 1
        # highcut = 45
        # order = 4
        # nyquist = 0.5 * sampling_rate
        # low = lowcut / nyquist
        # high = highcut / nyquist
        #
        # # Design Chebyshev bandpass filter
        # b, a = cheby1(order, 0.5, [low, high], btype='band')
        #
        # # Apply the filter using filtfilt to achieve zero-phase filtering
        # filtered_data = filtfilt(b, a, data, axis=1)
        #print(filtered_data.shape)
        signal_channels, signal_length = filtered_data.shape
        window_size = 250
        window_overlap = 0.5
        stride = int(window_size * (1 - window_overlap))
        num_windows = (signal_length - window_size) // stride + 1
        #print(num_windows)
        windows = np.zeros((signal_channels, num_windows, window_size))
        for j in range(num_windows):
            start = j * stride
            end = start + window_size
            windows[:, j, :] = filtered_data[:, start:end]
        labels = np.ones(windows.shape[1]) * label
        windows_with_labels = np.zeros(
            (signal_channels, num_windows, window_size + 1))  # Create an array of shape (16, 11, 501)

        for j in range(num_windows):
            start = j * stride
            end = start + window_size
            windows_with_labels[:, j, :window_size] = filtered_data[:, start:end]  # Copy values from the original array
            windows_with_labels[:, j, window_size] = labels[j]  # Set the label value for the corresponding window
        
        windows_with_labels = windows_with_labels.transpose(1, 0, 2)
        all_windows_with_labels.append(windows_with_labels)

# Convert the list of arrays to a single numpy array
all_windows_with_labels = np.concatenate(all_windows_with_labels, axis=0)
all_windows = torch.from_numpy(all_windows_with_labels)
# Define the CNN-MLP model

class CNN_MLP(nn.Module):
    def __init__(self, input_channels, input_size, hidden_size, output_size):
        super(CNN_MLP, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(128 * 60, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.norm = nn.BatchNorm1d(128)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print("222222", x.size())
        x = self.cnn(x)
        x = self.norm(x)
        # print('333333', x.size())
        x = self.flatten(x)
        x = self.mlp(x)
        return x

    # def preprocess_data(self, data):
    #     # Select the first 16 samples from the data
    #     data = data[0:16, :]
    #
    #     # Apply bandpass filter to the data
    #     sampling_rate = 128
    #     lowcut = 1
    #     highcut = 45
    #     order = 4
    #     nyquist = 0.5 * sampling_rate
    #     low = lowcut / nyquist
    #     high = highcut / nyquist
    #     b, a = signal.butter(order, [low, high], btype="band")
    #     filtered_data = signal.lfilter(b, a, data, axis=1)
    #
    #     return filtered_data
    # def predict(self, key):
    #     # Load the trained model parameters
    #     model_path = "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/model_state_dict.pth"
    #     model = CNN_MLP(16, 256, 1024, 4)
    #     model.load_state_dict(torch.load(model_path))
    #     model.eval()
    #
    #     # Preprocess the input data (similar to what you did in the training code)
    #     preprocessed_data = self.preprocess_data(key)
    #
    #     # Perform the prediction
    #     with torch.no_grad():
    #         inputs = torch.tensor(preprocessed_data, dtype=torch.float32)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         predicted_label = predicted.item()
    #
    #     return predicted_label

    # Prepare the data


data = all_windows[:, :, :-1]  # Your preprocessed data of size [176, 16, 500]
labels = all_windows[:, :, -1]  # Reshape labels to [2816]
labels = torch.tensor(labels[:, 0], dtype=torch.long)

# Perform train-test split
train_windows, test_windows, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Convert the training and testing data into PyTorch datasets
train_dataset = TensorDataset(
    torch.tensor(train_windows, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(test_windows, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)

# Create data loaders for batch processing
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the CNN-MLP model
input_channels = 16
input_size = 250
hidden_size = 1024
output_size = 4
model = CNN_MLP(input_channels, input_size, hidden_size, output_size)

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        #print('1111111111', inputs.size())
        optimizer.zero_grad()
        outputs = model(inputs)  # No need to transpose inputs anymore
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print the loss after each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# Testing loop
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        #print('11111111', inputs.shape)
        outputs = model(inputs)  # No need to transpose inputs anymore
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100}%")
save_path = '/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft_1/model_state_dict.pth'

# Make sure the directory exists before saving
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the model state dictionary
#torch.save(model.state_dict(), save_path)

# Save the model state dictionary
#torch.save(model.state_dict(), save_path)

