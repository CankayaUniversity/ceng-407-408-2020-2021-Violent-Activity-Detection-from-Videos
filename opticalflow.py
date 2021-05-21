import cv2 as cv
import numpy as np
import glob
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

magnitudeList = list()
orientationList = list()                    # Creating lists
magorientList = list()
magorientVector = list()
path ='D:\VADSVideos\Anomaly-Videos-Part-2\Fighting\*.*'    #Give a path for specific folder to read videos
video_list = []
VideoCounter = 0                                            #Video counter value holds number of videos


def videopath():
    global VideoCounter
    i = 0
    for video in glob.glob(path):                       #In this  code segment videos is sending one by one to opticalflow function
        a = cv.imread(video)
        video_list.insert(VideoCounter, video)
        VideoCounter = VideoCounter + 1
    while i != VideoCounter:
        print(video_list[i])
        opticalflow(video_list[i])
        i = i + 1


textfile = open("Magnitude.txt", "w")
textfile1 = open("Orientation.txt", "w")
textfile2 = open("Megorientation.txt", "w")
textfile3 = open("MagorientVector.txt", "w")
textfile4 = open("MagorientVectorNormal.txt","w")


def opticalflow(video):
    global magorientList
    capture = cv.VideoCapture(video)                #Capture the video
    ret, frame = capture.read()
    prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    #Getting frames from the video
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 244
    frameCounter = 0                                #frame counter holds the value that frame we have
    try:
        while (1):
            ret = capture.grab()  # grabing the frame
            frameCounter = frameCounter + 1  # increment counter
            if frameCounter % 10 == 2:  # display and detect only tenth of the frames,
                ret, frame = capture.retrieve()  # decoding the frame
                if ret:
                    ret2, frame2 = capture.read()
                    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0) #Calculating the opticalflow for each tenth frames
                    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])  # Extracting the attributes magnitude and angle (orientation)
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
                    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                    systemVertical = np.concatenate((frame2, rgb), axis=1)
                    cv.imshow('opticalflow', systemVertical)
                    mag = np.resize(mag, (224, 224, 3))  #Resize the magnitude and angle to function vGG16
                    ang = np.resize(ang, (224, 224, 3))
                    magnitudeList = np.asarray(mag)
                    orientationList = np.asarray(ang)
                    magorientList = mag * ang   # magOrientedList holds the value that multiplication of attributes(mag and orientation)
                    # print('MAGORIENTED: ', magorientList)
                    # transformedImage = np.expand_dims(magnitudeList, axis = 0)
                    # print(transformedImage.shape)
                    # print(magnitudeList)
                    textfile.write(str(mag))
                    textfile1.write(str(ang))           #Creating the text file for each attribute and their multiplication
                    textfile2.write(str(mag * ang))
                    prvs = next
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    return

    except:
        capture.release()
        cv.destroyAllWindows()

    # textfile.close()
    finally:
        vGG16(magorientList)  #Function call for vGG16


def vGG16(magorientList):
    global magorientVector

    model = VGG16(weights='imagenet', include_top=True) #model call vGG16 for transferLearning
    # for r in magorientList:
    transformedMatrices = np.expand_dims(magorientList, axis=0) #To add additional dimension for keras application
    print(transformedMatrices.shape)
    transformedMatrices = preprocess_input(transformedMatrices)
                                                           #preprocess_input function properly transforms a standard matrice into the format which model requires.
    print(transformedMatrices)                             #print matrices
    prediction = model.predict(transformedMatrices)        #predict() function classify input matrices in 1000 possible classes.
    magorientVector.append(prediction)                     #Append the vetor list to magorientVector
    textfile3.write(str(magorientVector))                  #Writing the textfile
    print('prediction', prediction)
    print(prediction.shape)
    # predictionLabel = decode_predictions(prediction, top=5)
    # print(predictionLabel)
    # print first prediction probality
    # print('%s (%.2f%%)' % (predictionLabel[0][0][1], predictionLabel[0][0][2] * 100))

videopath()