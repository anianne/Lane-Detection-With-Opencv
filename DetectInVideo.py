import numpy as np
import cv2 as cv

#cap = cv.VideoCapture('lane1.mp4')
frame = cv.imread('Capture.jpg')
#while(cap.isOpened()):
for j in range(1):
    #ret, frame = cap.read()
    #print(frame.shape)
    img = frame
    image = img
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv.bitwise_and(edges, mask)
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    linelist = []
    highx=[700,00]
    highy=[00,00]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            #cv.line(img,(x1,y1),(x2,y2),(255,0,0),10)
        linelist.append([x1,y1])
        linelist.append([x2,y2])
    color_edges = np.dstack((edges, edges, edges)) 
    lines_edges = cv.addWeighted(color_edges, 0.8, line_image, 1, 0)
    for line in range(len(linelist)):
        x1 = linelist[line][0]
        y1 = linelist[line][1]
        cv.putText(lines_edges,str((x1,y1)),(x1,y1),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        if x1 < highx[0] and y1 > highy[1]:
                highx[0] = x1
                highy[0] = y1
                #highx[0] = y1
                #highx[1] = x1
    for line in range(len(linelist)):
        x1 = linelist[line][0]
        y1 = linelist[line][1]
        if y1 == highy[0] and x1 != highx[0]:        
            if x1 > highx[0] and y1>highy[1]:
                highx[1] = x1
                highy[1] = y1
                #highy[0] = y1
                #highy[1] = x1
    print(highx,highy)
    cv.putText(lines_edges,str((highx[0],highx[1])),(highx[0],highx[1]),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv.line(lines_edges,(int((highx[1]+highy[1])/2),0),(int((highx[1]+highy[1])/2),540),(255,0,0),10)
    cv.line(img,((highx[0]+highy[0])/2,0),((highx[0]+highy[0])/2,540),(255,0,0),10)       
    cv.line(img,(475,0),(475,540),(1255,255,0),10)
    cv.line(lines_edges,(475,0),(475,540),(255,255,0),10)
    if (((highx[0]+highy[0])/2) - 475) > 50:
        color = (0,0,225)
        txtpt = 'turn left'
    elif (((highx[0]+highy[0])/2) - 475) < -50:
        txtpt = 'turn right'
        color = (0,0,225)
    else:
        color = (0,225,0)
        txtpt = 'straight'

    cv.putText(lines_edges,txtpt,(25,25),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2) 
    #cv.rectangle(lines_edges,((highx[0]+highy[0])/2,0),(475,540),(color),-1)
    cv.putText(img,txtpt,(25,25),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2) 
    #cv.rectangle(img,((highx[0]+highy[0])/2,0),(475,540),(color),-1)
    cv.imshow('img5',lines_edges)
    #cv.imshow('img4',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
