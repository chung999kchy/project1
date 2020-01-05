# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:03:34 2019

@author: chung
"""

import os
import cv2
import numpy as np
import calOptical as AD
INPUT_VIDEO = 'cam2.mp4'
OUTPUT_IMG = 'out_my_video'
os.makedirs(OUTPUT_IMG, exist_ok=True)

def print_image(img, frame_diff):
    new_img = np.zeros([img.shape[0], img.shape[1]*2, img.shape[2]]) # [height, width*2, channel]
    new_img[:, :img.shape[1], :] = img         # place color image on the left side
    new_img[:, img.shape[1]:, :] = frame_diff  # place gray image on the right side
    return new_img

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    B=1
    print("width:", frame_width)
    print("height:", frame_height)
    TOng=B*frame_width*frame_height
    print(TOng)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('out_diff1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    out1 = cv2.VideoWriter('out_op1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    idx = -1
    _, last_frame=cap.read()
    while(True):
        ret, frame = cap.read()
        idx += 1
        if not ret:
            print('Stopped reading the video (%s)' % video_path)
            break

        diff = cv2.absdiff(frame, last_frame)
        if (diff.sum() > TOng):
            check=AD.calOp(last_frame,frame,frame_width,frame_height)
            if check==True:
                #mix=print_image(frame, diff)
                #cv2.imwrite(os.path.join(OUTPUT_IMG, 'img_%06d.jpg' % idx),mix )
                out.write(diff)
                out1.write(frame)
                cv2.imshow("diff",diff)
        last_frame=frame
        key=cv2.waitKey(30)
        if key==27:
            break
    out.release()
    out1.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print('Running frame difference algorithm on %s' % INPUT_VIDEO)
    main(video_path=INPUT_VIDEO)
