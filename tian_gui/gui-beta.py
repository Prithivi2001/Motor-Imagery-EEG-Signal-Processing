# -*-coding:utf-8-*-
import tkinter as tk
from tkinter import *
from video_player import tkvideo
from PIL import Image, ImageTk
import threading


#Importing Videos
video_list = ['legs.mp4', 'right.mp4', 'left.mp4']
text_list = ['Riding', 'Right Hand Eating', 'Left Hand Wiping']
current = 0

def switch_video(step, label):

    global current, video_list
    current = (current + step) % len(video_list)
    video = video_list[current]

    player = tkvideo(video, label, loop = 1, size = (1280,720), hz = 30, elapse = 10)
    label['text'] = text_list[current]
    player.play()




class MainWindow:

    def __init__(self):

        self.app = tk.Tk(className='Data Gathering System')
        self.app.geometry('1400x850+245+110')
        self.app.title('Data Gathering Application')
        self.frm1 = Frame(self.app)
        self.createpage()


    def createpage(self):

        self.frm1.config(height=720, width=1280)
        self.video_label = Label(self.app)
        self.video_label.place(x=60, y=30)


        Button(self.app, text='Start', command=lambda: switch_video(0, self.video_label), height=2, width=15).place(x=350, y=780)
        Button(self.app, text='Previous', command=lambda: switch_video(-1, self.video_label), height=2, width=15).place(x=550, y=780)
        Button(self.app, text='Next', command=lambda: switch_video(+1, self.video_label), height=2, width=15).place(x=750, y=780)
        Button(self.app, text='Exit', command=self.app.quit, height=2, width=15).place(x=950, y=780)

        # switch_video(0, self.video_label)


if __name__ == '__main__':
    MainWindow()
    mainloop()
