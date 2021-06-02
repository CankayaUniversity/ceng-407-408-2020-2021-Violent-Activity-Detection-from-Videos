#This code has been  run  for different  'normal' and 'fight' videos dataset

import cv2 as cv
import numpy as np
import glob
from numpy import savetxt


path = ('/path')  #Give a path for specific folder to read videos
video_list = []
VideoCounter = 0  #Video counter value holds number of videos
videoid2 = 1
videoid = 0
sample = 0
sample2 = 0
sequence_arr = []
labelArray = []


def videopath():                 #In this  code segment videos is sending one by one to video frame function
    global VideoCounter
    i = 0
    for video in glob.glob(path):
        video_list.insert(VideoCounter, video)
        VideoCounter = VideoCounter + 1
    while i != VideoCounter:
        video_frame(video_list[i])
        sequence_arr.append([video_list[i], sample2])  #sample2 holds the number of sequences. Every turnaround, append the number of samples the sequence array
        np.savetxt('Data_Sequence.txt', sequence_arr, fmt='%s')
        i = i + 1


def video_frame(video):
    global videoid
    print(video)
    capture = cv.VideoCapture(video)   #Capture the video
    ret, frame = capture.read()        #Getting frames from the video
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 244
    frameCounter = 0    #frame counter holds the value that frame we have
    counter = 0  # video counter
    videoid = videoid + 1
    try:
        while (1):
            ret = capture.grab()  # grabing the frame
            frameCounter = frameCounter + 1  # increment counter
            if frameCounter % 5 == 2:  # display and detect only fifth of the frames,
                ret, frame = capture.retrieve()  # decoding the frame
                if ret:       #If the ret is true, video and videoid will send the video_frame Function
                    ret2, frame2 = capture.read()   #video capture for the second frame
                    video_label(video, videoid)   #call the video_label function

                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    return

    except:
        capture.release()


def video_label(video, videoid): #videoid holds the number of videos
    global videoid2
    global sample      # sample holds the number of frames
    global sample2  # sample2 holds the number of sequence (30)
    sample = sample + 1
    video_name = video
    if (videoid == videoid2):
        if (sample % 30 == 0):  #
            labelArray.append([video_name, 1])  # 1 for fight, 0 for normal
            label_arr = np.asarray(labelArray)   #every sequence append the label value 0 for normal videos 1 for fight videos
            np.savetxt('Data_Label_Fight.txt', label_arr, fmt='%s')  #saving txt file for every label sequence
            sample2 = sample2 + 1

    else:
        sample = 0  # assign sample the '0' value to pass the next video
        sample2 = 0
        videoid2 = videoid   #assign videoid2 to next video

videopath()