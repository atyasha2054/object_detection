import cv2
import os
import numpy as np
import seaborn as sb
import pandas as pd
from matplotlib import pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox




plot1 = plt.subplot2grid((2, 2), (0, 0))
plot2 = plt.subplot2grid((2, 2), (1, 0), rowspan=2)


vid=cv2.VideoCapture(0)

labels=[]

#for i in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
while True:

    success,frame=vid.read()
    frame= cv2.resize(frame, (960, 540))
    bbox,label,conf=cv.detect_common_objects(frame)
    output_image=draw_bbox(frame,bbox,label,conf)

    cv2.imshow("Object Detection",output_image)

    for item in labels:
        if item in labels:
            pass
        else:
            labels.append(item)


    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

vid.release()
cv2.destroyAllWindows
print(labels)


