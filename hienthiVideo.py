# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:33:02 2019

@author: chung
"""
import numpy as np
import cv2

cap = cv2.VideoCapture('out_op1.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('output',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()