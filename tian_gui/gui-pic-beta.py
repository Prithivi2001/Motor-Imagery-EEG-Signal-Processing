# -*-coding:utf-8-*-
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


#Importing Videos
image_list = ['legs.png', 'right.png', 'left.png']
current = 0

def switch_pic(step, pic_label):
    global current, image_list, text_list

    current = (current + step) % len(image_list)
    image = Image.open(image_list[current]).resize((1280, 720))
    photo = ImageTk.PhotoImage(image)
    pic_label['image'] = photo
    pic_label.photo = photo




class MainWindow:

    def __init__(self):

        self.app = tk.Tk(className='Data Gathering System')
        self.app.geometry('1400x850+245+110')
        self.app.title('Data Gathering Application')
        self.frm1 = Frame(self.app)
        self.createpage()


    def createpage(self):

        self.frm1.config(height=720, width=1280)
        self.pic_label = Label(self.app)
        self.pic_label.place(x=60, y=30)


        Button(self.app, text='Start', command=lambda: switch_pic(0, self.pic_label), height=2, width=15).place(x=350, y=780)
        Button(self.app, text='Previous', command=lambda: switch_pic(-1, self.pic_label), height=2, width=15).place(x=550, y=780)
        Button(self.app, text='Next', command=lambda: switch_pic(+1, self.pic_label), height=2, width=15).place(x=750, y=780)
        Button(self.app, text='Exit', command=self.app.quit, height=2, width=15).place(x=950, y=780)

        # switch_video(0, self.video_label)


if __name__ == '__main__':
    MainWindow()
    mainloop()
