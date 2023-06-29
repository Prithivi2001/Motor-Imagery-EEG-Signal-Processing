#This script handles the connection of the initial data that is collected from participants 
#which is passed to our model based on 3 second segments to generate prediction values 
#passed to car_control_pipeline.py
import sys
import tty
import termios
import select
import socket
import curses
import os
import time
import torch
import numpy as np
from distutils.command.config import config
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from new_model import EEGPredictor
from model import XXXPNet_Basic



write_path = "/tmp/command"
wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)


BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
# for ubuntu
params.serial_port = '/dev/ttyUSB0'

board = BoardShim(2, params)
board.prepare_session()
board.start_stream()

#wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

localPort = 1234
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
print("UDP server up and listening")


model_path = "/home/julius/Documents/GitHub/research/EEG_controller_data_record/EEG/scripts/car_infer/Net/Tensor_CSPNet_model.pth"
model = XXXPNet_Basic()
model.load_state_dict(torch.load(model_path))
model.eval()

eeg_predictor = EEGPredictor()
eeg_predictor.set_model(model)

def save_file(key_data, file_name):
     file_path = os.path.join("/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft",file_name)
     directory = os.path.dirname(file_path)
     if not os.path.exists(directory):
         os.makedirs(directory)
    
     with open(file_path, "wb") as f:
         np.save(f, key_data, allow_pickle=False, fix_imports=False)

#data_queue = queue.Queue()


def getKey():
     for val in board.get_board_data():
         #data_file_num = str(file_num) + ".npy"
         time.sleep(3)
         key = np.array(board.get_board_data())
         key = key[1:17, :]
         if key.shape[1] != 384:
            diff = 384 - key.shape[1]
            new_arr = np.repeat(key[:, -1:], diff, axis=1)
            key = np.concatenate((key, new_arr), axis=1)  
            
         prediction = eeg_predictor.predict(key)
         print('prediction:', prediction)
       #prediction values : 'w' : Forward, 's' : Backward, 'a' : Left, 'd' : Right, 'z' : Stop
         if prediction == 0:
            return 'w'
         elif prediction == 1:
            return 's'
         elif prediction == 2: 
            return 'a'
         elif prediction == 3: 
            return 'd'
         elif prediction == 4: 
            return 'z'
                
stdscr = curses.initscr()
curses.cbreak()

try:
    while True:
        key = getKey()
        if key is not None:
            if key == 'q':
                print("exit!")
                break
            print(key)
            msg = key.encode('ascii')
            len_send = os.write(wf, msg)

finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()
    os.write(wf, 'exit'.encode('ascii'))
    os.close(wf)
