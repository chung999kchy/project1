import numpy as np
import cv2
import math
import statistics
#Create K is value of displacement Li
K=1
K1=0.001
def displacement(a,b,c,d):
    dic=(a-c)**2+(b-d)**2
    return math.sqrt(dic)

#params for ShiTomasi corner dectection
feature_params=dict(maxCorners=1000,
                    qualityLevel=0.3,
                    minDistance=2,
                    blockSize=7)

#Params for Lucas Kanade optical flow
lk_params=dict( winSize=(15,15),
               maxLevel=4,
               criteria=( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Create some random colors
color =np.random.randint(0,255,(100,3))

#Take first frame and find corners in it
def calOp (last_frame, frame,frame_width,frame_height):
    # Define the codec and create VideoWriter object.The output is stored in 'output3.avi' file.
    
    old_gray=cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    p0=cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    #mask=np.zeros_like(last_frame)
    frame_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #calculate optical flow
    p1, st, err=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #Select good points
    good_new=p1[st==1]
    good_old=p0[st==1]
    check=False
    mang=[]
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b=new.ravel()
        c,d=old.ravel()      
        Li=displacement(a,b,c,d)
        if (Li>K):
            mang.append(Li)
    if len(mang) >= 2:
        TrungBinh=statistics.mean(mang)
        DoLechChuan=statistics.stdev(mang)
        if DoLechChuan/TrungBinh > K1:
            check=True
    return check
    
            