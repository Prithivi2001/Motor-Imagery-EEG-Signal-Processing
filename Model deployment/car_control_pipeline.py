#This script handles the movement of the 4WD Robot Car through ascii commands it receives while receiving
#commands from our personal server which is connected to the EEG Cap and the proposed model 
import os
import numpy as np
import socket
import time


# function to send UDP command, port number 1234
def udp_command_sender(command):
    UDPServerSocket.sendto(str.encode(
        command), ('192.168.117.117', 1234))


def to_command(argument): # default value is [S]
    switcher = {
        'w': "[F]",
        's': "[B]",
        'a': "[L]",
        'd': "[R]",
        'z': "[S]"
    }
    return switcher.get(argument, "[S]")

def pipe_data():
    global key
    try:
        key = os.read(rf, 8)
        key = key.decode("utf-8")
        

    except OSError as e:
        if e.errno == 11:
            print("wait")
        else:
            print("something wrong")


# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
print("UDP ready")



read_path = "/tmp/command"
if os.path.exists(read_path):
    os.remove(read_path)
os.mkfifo(read_path)
rf = os.open(read_path, os.O_NONBLOCK | os.O_RDONLY)

while True:
    
    pipe_data()
    if key == 'exit':
        print("received msg:", key, "terminate.")
        break
    udp_command_sender(to_command(key))
    

       
os.close(rf)

