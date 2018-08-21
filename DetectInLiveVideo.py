import numpy as np
import cv2 as cv
errorfile = open('error.txt','a+')
#cap = cv.VideoCapture('lane1.mp4')
cap = cv.VideoCapture(0)
#while(cap.isOpened()):
while True:
    ret, frame = cap.read()
    img = frame
    image = img
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
# Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)
# Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)
# Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv.bitwise_and(edges, mask)
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    linelist = []
    highx=[700,00]
    highy=[00,0]
# Iterate over the output "lines" and draw lines on a blank image
    if  str(lines) != 'None':
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            linelist.append([x1,y1])
            linelist.append([x2,y2])
# Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
        lines_edges = cv.addWeighted(color_edges, 0.8, line_image, 1, 0)
        for line in range(len(linelist)):
            x1 = linelist[line][0]
            y1 = linelist[line][1]
    #x2 = linelist[line-1][2]
    #y2 = linelist[line-1][3]    
        #cv.putText(lines_edges,str((x1,y1)),(x1,y1),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
    #print(x1,y1,x2,y2)
            if x1 < highx[0] and y1 >highx[1]:
                highx[0]=x1
                highx[1]=y1
            elif x1 > highy[0] and y1 > highy[1]:
                highy[0] = x1
                highy[1] = y1
        cv.putText(lines_edges,str((highx[0],highx[1])),(highx[0],highx[1]),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv.putText(lines_edges,str((highy[0],highy[1])),(highy[0],highy[1]),cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        if (((highx[0]+highy[0])/2) - 475) > 50:
            color = (0,0,225)
            txtpt = 'turn left'
            errorfile.write(str([int((highx[0]+highy[0])/2),int((highx[0]+highy[0])/2-475),highx,highy]))
            print(str([int((highx[0]+highy[0])/2),int((highx[0]+highy[0])/2-475),highx,highy]))
            errorfile.write(str('\n'))
        elif (((highx[0]+highy[0])/2) - 475) < -50:
            txtpt = 'turn right'
            color = (0,0,225)
        #errors.append(str([int(highx[0]+highy[0]),int((highx[0]+highy[0])/2),int((highx[0]+highy[0])/2-475),highx,highy]))
            print(str([int((highx[0]+highy[0])/2),int((highx[0]+highy[0])/2-475),highx,highy]))
            errorfile.write(str([int((highx[0]+highy[0])/2),int((highx[0]+highy[0])/2-475),highx,highy]))
            errorfile.write(str('\n'))
        else:
            color = (0,225,0)
            txtpt = 'straight'
    #print(txtpt)
    #print(highx,highy)
    #print((highx[1]+highy[0])/2)
        #cv.line(lines_edges,(475,0),(475,540),(255,255,0),10) #light blue-Center of screen
        #cv.rectangle(lines_edges,((highx[0]+highy[0])/2,0),(475,540),color,-1)
        cv.line(lines_edges,((highx[0]+highy[0])/2,0),((highx[0]+highy[0])/2,540),(255,0,0),10) #Dark blue-Lane Center
        #cv.line(img,(475,0),(475,540),(255,255,0),10) #light blue-Center of screen
        #cv.rectangle(img,((highx[0]+highy[0])/2,0),(475,540),color,-1)
        cv.line(img,((highx[0]+highy[0])/2,0),((highx[0]+highy[0])/2,540),(255,0,0),10) #Dark blue-Lane Center
        cv.imshow('img5',lines_edges)
        cv.imshow('img4',img)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
#print(errors)
errorfile.close()
