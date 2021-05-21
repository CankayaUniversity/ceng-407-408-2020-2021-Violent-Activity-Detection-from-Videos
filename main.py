import tkinter as tk
from tkinter import *
import cv2 as cv
import numpy as np


def onclick():
    capture1 = cv.VideoCapture("Videos/1.mp4", 0)
    capture2 = cv.VideoCapture("Videos/2.mp4", 0)
    capture3 = cv.VideoCapture("Videos/3.mp4", 0)          #Video capture from videos folder
    capture4 = cv.VideoCapture("Videos/4.mp4", 0)
    capture5 = cv.VideoCapture("Videos/5.mp4", 0)
    capture6 = cv.VideoCapture("Videos/6.mp4", 0)

    while True:
        ret1, cam1 = capture1.read()
        cam1 = cv.resize(cam1, (640, 540),)
        ret2, cam2 = capture2.read()
        cam2 = cv.resize(cam2, (640, 540),)
        ret3,  cam3 = capture3.read()
        cam3 = cv.resize(cam3, (640, 540),)                 #Resizing every video to 640, 540
        ret4, cam4 = capture4.read()
        cam4 = cv.resize(cam4, (640, 540),)
        ret5, cam5 = capture5.read()
        cam5 = cv.resize(cam5, (640, 540),)
        ret6, cam6 = capture6.read()
        cam6 = cv.resize(cam6, (640, 540))


        if ret1 == True:
            systemVertical = np.concatenate((cam1, cam2, cam3), axis=1)#Concatenate the first three videos in vertical with numpy
            systemSecondVertical = np.concatenate((cam4, cam5, cam6), axis=1)#Concatenate the second three videos in vertical with numpy

            system = np.concatenate((systemVertical, systemSecondVertical), axis=0)#Concatenate the videos horizontally with numpy
            cv.imshow('VADS', system) #imshow the system
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    capture1.release()
    cv.destroyAllWindows()


def MainScreen():

    screen = tk.Tk()
    screen.title("VADS")
    screen.geometry("1920x1080")           #Creating the screen with tk
    #img=cv.imread('images/surveillance.jpg')
    #cv.imshow("output", img)
    #cv.imwrite("surveillance.jpg", img)

    #Image
    bg = PhotoImage(file='images/surveillance.gif')     #Taken background image for images folder
    screen_canvas = Canvas(screen, width=1920, height=1080)
    screen_canvas.pack(fill="both", expand=True)

    #Image set
    screen_canvas.create_image(0, 0, image=bg, anchor="nw")

    screen_canvas.create_text(960, 350,
                              text="Welcome to ",                            #Adjusting the MainScreen
                              font=("Times", 40), fill="lightgray")

    screen_canvas.create_text(960, 450,
                              text="Violent Activity Detection System ",
                              font=("Times", 40), fill="lightgray")
    screen_canvas.create_text(960, 550,
                              text="Please press 'Start' button to run system",
                              font=("Times", 20), fill="lightgray")
    start_button = Button(screen, text="START", command=onclick)#every time the user push the button call the onclick function
    start_button_window = screen_canvas.create_window(960, 650, anchor="nw", window=start_button)

    '''
    #Screen Label
    screen_label = Label(screen, image=bg)
    screen_label.place(x=0, y=0, relwidth=1, relheight=1)

    screen_text = Label(screen, text="Welcome!", font=("Helvetica,",50),fg="white")
    screen_text.pack(pady=50)

    start_button = Button(screen, text="Start")
    start_button.pack(pady=20)
    '''

    screen.mainloop()
MainScreen()