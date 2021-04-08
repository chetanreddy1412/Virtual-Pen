import cv2
import numpy as np


def get_pen_hsv_val():
    
    lower = np.array([90,54,0])
    upper = np.array([131,255,255])
        
    cap = cv2.VideoCapture(0)

    def nothing(pos):
        pass

    trackbarwindow = 'Adjusting the Pen Color'
    
    cv2.namedWindow(trackbarwindow)
    cv2.createTrackbar('lh',trackbarwindow,lower[0],179,nothing)
    cv2.createTrackbar('uh',trackbarwindow,upper[0],179,nothing)
    cv2.createTrackbar('ls',trackbarwindow,lower[1],255,nothing)
    cv2.createTrackbar('us',trackbarwindow,upper[1],255,nothing)
    cv2.createTrackbar('lv',trackbarwindow,lower[2],255,nothing)
    cv2.createTrackbar('uv',trackbarwindow,upper[2],255,nothing)
    cv2.createTrackbar('save',trackbarwindow,0,1,nothing)


    while cap.isOpened():
    
        ret, frame = cap.read()
        frame = cv2.flip(frame,1) #flipping the video feed
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        lh = cv2.getTrackbarPos('lh',trackbarwindow)
        uh = cv2.getTrackbarPos('uh',trackbarwindow)
        ls = cv2.getTrackbarPos('ls',trackbarwindow)
        us = cv2.getTrackbarPos('us',trackbarwindow)
        lv = cv2.getTrackbarPos('lv',trackbarwindow)
        uv = cv2.getTrackbarPos('uv',trackbarwindow)
        save = cv2.getTrackbarPos('save',trackbarwindow)
    
        lower = np.array([lh,ls,lv])
        upper = np.array([uh,us,uv])
        mask = cv2.inRange(hsv,lower,upper)
    
        kernel = np.ones((5,5),np.uint8)
        mask_eroded = cv2.erode(mask,kernel,iterations=1)
        mask= cv2.dilate(mask_eroded,kernel,iterations = 2)
    
        frame_with_mask = cv2.bitwise_and(frame,frame,mask=mask)
        mask_3d = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((mask_3d,frame))

        cv2.imshow(trackbarwindow,cv2.resize(stacked,None,fx=0.7,fy=0.7))

        k = cv2.waitKey(1)
        if save == 1:
            print([lower,upper])
            break
        if k == 27 or k == 13:
            break

    
    cv2.destroyAllWindows()
    cap.release()
    penval = [lower,upper]
    return penval
        
def get_pen_coordinates(frame,pen_hsv):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,*pen_hsv)
            
    kernel = np.ones((5,5),np.uint8)
    mask_eroded = cv2.erode(mask,kernel,iterations=1)
    mask= cv2.dilate(mask_eroded,kernel,iterations = 2)
    
    contours,h = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_area_cnt = max(contours,key = cv2.contourArea)
        if cv2.contourArea(max_area_cnt)>100:
            x,y,w,h = cv2.boundingRect(max_area_cnt)
            return (int(x+w/2),int(y+h/2))
                
def main():
    pen_hsv = get_pen_hsv_val()
    
    cap = cv2.VideoCapture(0)
    pt1 = (-1,-1)
    pause = False
    pt2 = (-1,-1)
    
    ret,frame_i = cap.read()
    drawing_sheet = None
    points = []
    
    while cap.isOpened():
        

        if drawing_sheet is None:
            drawing_sheet = np.zeros_like(frame_i)            

        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        #frame = frame_orig.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,*pen_hsv)
        
        kernel = np.ones((5,5),np.uint8)
        mask_eroded = cv2.erode(mask,kernel,iterations=1)
        mask= cv2.dilate(mask_eroded,kernel,iterations = 2)
        
        pen_coordinates = get_pen_coordinates(frame,pen_hsv)
        #print(pen_coordinates)
        #cv2.line(frame,(0,0),frame.shape[:2],[200,50,50],4)
        if pen_coordinates != None and pause != True:
            #print(i)
            if pt2 != (-1,-1):
                
                pt1 = pt2
                pt2 = pen_coordinates
                #print(pt1)
                #print(pt2)
                points.append(pt1)
                drawing_sheet = cv2.line(drawing_sheet,pt1,pt2,[200,50,50],4)
                
            elif pt2 == (-1,-1):
                pt2 = pen_coordinates
                
        else:
            pt1 = (-1,-1)
            pt2 = (-1,-1)        
        frame = cv2.add(frame,drawing_sheet)
        cv2.imshow('Video Feed',frame)
        cv2.imshow('Drawing Board',drawing_sheet)
        #cv2.imshow('mask',mask)
        k = cv2.waitKey(10)
        if k == 27:
            break
            
        if k == ord('c'):
            drawing_sheet = np.zeros_like(frame_i)

        if k == ord('h'):
            pause = True
        else:
            pause = False
            
    
    cv2.destroyAllWindows()
    cap.release()
    
if __name__ == '__main__':
    main()
