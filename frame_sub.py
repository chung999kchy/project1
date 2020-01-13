# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:03:34 2019

@author: chung
"""

import os
import cv2
import numpy as np
import calOptical as AD
INPUT_VIDEO = 'chess5.mp4'
OUTPUT_IMG = 'out_my_video'
os.makedirs(OUTPUT_IMG, exist_ok=True)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    B=1
    print("frame_width:", frame_width)
    print("frame_height:", frame_height)
    TOng=B*frame_width*frame_height
    print("K_sub", TOng)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
   
    out1 = cv2.VideoWriter('out_op5.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    _, last_frame=cap.read()
    while(True):
        ret, frame = cap.read()

        if not ret:
            print('Stopped reading the video (%s)' % video_path)
            break

        diff = cv2.absdiff(frame, last_frame)
        if (diff.sum() > TOng):
            check=AD.calOp(last_frame,frame,frame_width,frame_height)
            if check==True:
                out1.write(frame)
                cv2.imshow("frame",frame)
                last_frame=frame
        key=cv2.waitKey(30)
        if key==27:
            break
    out1.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print('Running frame difference algorithm on %s' % INPUT_VIDEO)
    main(video_path=INPUT_VIDEO)
