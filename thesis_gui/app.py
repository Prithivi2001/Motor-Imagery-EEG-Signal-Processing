import PIL
import PIL.Image
from PIL import ImageTk
from tkinter import *


root = Tk()
root.geometry("1500x800")
"""
running_prog = TRUE
starting_sec = 0
ending_sec = 10
image_arr = ["picture_2.jpg", "picture_3.jpg"]
image_count = 0


def photocountdown():
    global running_prog, starting_sec, ending_sec, image_arr, image_count
    while(running_prog):
        time.sleep(1)
        starting_sec += 1
        if(starting_sec >= ending_sec and image_count <= len(image_arr)):
            running_prog = False
            photo_label = ImageTk.PhotoImage(
                PIL.Image.open(image_arr[image_count]))
            starting_sec = 0
            running_prog = True
        else:
            permphoto_label = ImageTk.PhotoImage(PIL.Image.open(image_arr[-1]))
            permphoto_description = Label(
                root, text="EEG Signal experimentation finish")
            running_prog = False

"""
#image_arr = ["picture_2.jpg", "picture_3.jpg", "picture_4.jpg"]
image_1_open = ImageTk.PhotoImage(PIL.Image.open("images/picture_1.png"))
image_2_open = ImageTk.PhotoImage(PIL.Image.open("images/picture_2.png"))
image_3_open = ImageTk.PhotoImage(PIL.Image.open("images/picture_3.png"))
#image_4_open = ImageTk.PhotoImage(PIL.Image.open("images/picture_4.png"))

image_func_arr = [image_1_open, image_2_open, image_3_open]

new_label = Label(image=image_1_open)
new_label.grid(row=0, column=0, columnspan=3)


def increase(image_num):
    global new_label, button_more, button_less
    new_label.grid_forget()
    new_label = Label(image=image_func_arr[image_num - 1])
    button_more = Button(
        root, text=">", command=lambda: increase(image_num + 1))
    button_less = Button(
        root, text="<", command=lambda: decrease(image_num - 1))

    if(image_num == len(image_func_arr)):
        button_more = Button(root, text=">", state=DISABLED)

    new_label.grid(row=0, column=0, columnspan=3)
    button_less.grid(row=1, column=0)
    button_more.grid(row=1, column=2)


def decrease(image_num):
    global new_label, button_more, button_less
    new_label.grid_forget()
    new_label = Label(image=image_func_arr[image_num - 1])
    button_more = Button(
        root, text=">", command=lambda: increase(image_num + 1))
    button_less = Button(
        root, text="<", command=lambda: decrease(image_num - 1))

    if(image_num == 1):
        button_less = Button(root, text="<", state=DISABLED)

    new_label.grid(row=0, column=0, columnspan=3)
    button_less.grid(row=1, column=0)
    button_more.grid(row=1, column=2)


button_less = Button(root, text="<", command=decrease, state=DISABLED)
button_finish = Button(root, text="Finish Program", command=root.quit)
button_more = Button(root, text=">", command=lambda: increase(2))
welcome_label = Label(root, text="EEG Signal Collection Images")


button_less.grid(row=1, column=0)
button_finish.grid(row=1, column=1)
button_more.grid(row=1, column=2)


# welcome_label.pack()


# Event Loop

root.mainloop()
