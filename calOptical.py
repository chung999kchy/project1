import numpy as np
import cv2
import math
#Create K is value of displacement Li
K=3
K_tong=6
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
    mask=np.zeros_like(last_frame)
    frame_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #calculate optical flow
    p1, st, err=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    #Select good points
    good_new=p1[st==1]
    good_old=p0[st==1]
    check=False
    tong=0
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b=new.ravel()
        c,d=old.ravel()
        e=int(4*c-3*a)
        f=int(4*d-3*b)        
        Li=displacement(a,b,c,d)
        if (Li>K):
            tong+=Li
            mask = cv2.line(mask, (a,b),(e,f), color[22].tolist(), 1)
            frame = cv2.circle(frame,(c,d),2,color[34].tolist(), -1)
            frame=cv2.add(frame,mask)
            
    if tong > K_tong:
        cv2.imshow('frame',frame)
        check=True
            
    return check
    
            