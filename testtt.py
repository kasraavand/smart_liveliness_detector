import cv2
import sys
import numpy as np
from collections import deque, defaultdict
from functools import partial



faceCascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")
video_capture = cv2.VideoCapture(0)

video_capture.set(3,640)
video_capture.set(4,480)

SF = 1.05
coordination_change_faactor = 50
frame_series = defaultdict(partial(deque, maxlen=20))


class PositionFactor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ccf = coordination_change_faactor 

    def __eq__(self, arg):
        x, y = arg
        return (self.x - self.ccf <= x <= self.x + self.ccf) and \
    (self.y - self.ccf <= y <= self.y + self.ccf) 
    
    def __hash__(self):
        return 1
    
    def __repr__(self):
        return "{}_{}".format(self.x, self.y)

    def __iter__(self):
        return iter((self.x, self.y))



def cal_smile(crop):
    smile = smileCascade.detectMultiScale(
                crop,
                scaleFactor= 1.2,
                minNeighbors=55,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
    try:
        return smile[0]
    except IndexError:
        return np.array([])

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]

        key = PositionFactor(x, y)
        frame_series[key].append(roi_gray)
        
        roi_color = frame[y:y+h, x:x+w]
        
        smile = cal_smile(roi_gray)
        # print(smile)
        try:
            (x, y, w, h) = smile
        except:
            # no face detected
            pass
        else:
            if len(frame_series[key]) == 20:
                print("stable face")
                #cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
                s = sum(cal_smile(j).any() for j in frame_series[key])
                print(s)
                if s < 10:
                    cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    print(key)
                    # frame_series[key].clear()

        previous_face = roi_gray
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()