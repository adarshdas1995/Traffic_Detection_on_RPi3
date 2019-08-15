#!/usr/bin/env python
# coding: utf-8

# In[15]:


#import libraries of python opencv
import cv2
import numpy as np
import time
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
#import imutils


# In[9]:


def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            frame = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(frame, 0.007843,(300, 300), 127.5)
            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()
            # write the detections to the output queue
            outputQueue.put(detections)


# In[4]:


CLASSES = [ "background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# In[5]:


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


# In[7]:


# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None


# In[11]:


# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,outputQueue,))
p.daemon = True
p.start()


# In[23]:


#create VideoCapture object and read from video file
cap = cv2.VideoCapture('cars.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
fps = FPS().start()


# In[24]:


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it, and
    # grab its dimensions
    ret,frame = cap.read()
    try:
        frame = cv2.resize(frame, (400,400))
    except:
        break
    (fH, fW) = frame.shape[:2]
    # if the input queue *is* empty, give the current frame to classify
    if inputQueue.empty():
        inputQueue.put(frame)
    # if the output queue *is not* empty, grab the detections
    if not outputQueue.empty():
        detections = outputQueue.get()
    if detections is not None:
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.2:
                continue
            idx = int(detections[0, 0, i, 1])
            dims = np.array([fW, fH, fW, fH])
            box = detections[0, 0, i, 3:7] * dims
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    #display the resulting frame
    cv2.imshow('video', frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break  
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#release the videocapture object
cap.release()
cv2.destroyAllWindows()

