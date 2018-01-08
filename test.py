import numpy as np
import scipy.ndimage
import cv2
from matplotlib import pyplot as plt
import math

def nothing(x):
    pass

def findBoundaryCorners(blackAndWhite):
    height, width = blackAndWhite.shape
    edgeWidth = int(width*0.15)
    edgeHeight = int(height*0.15)
    if width < edgeWidth or height < edgeHeight:
        return []
    innerImg = np.zeros((height+6, width+6), np.uint8)
    innerImg[3:-3,3:-3] = blackAndWhite
    corners = np.ones(blackAndWhite.shape,np.uint8)*21
    for ii in range(1,6):
        corners -= innerImg[0:-6,ii:(ii-6)]
        corners -= innerImg[6: , ii:(ii-6)]
        corners -= innerImg[ii:(ii-6),0:-6]
        corners -= innerImg[ii:(ii-6),6:  ]
    # corners = (corners*innerImg[3:-3,3:-3])
    corners = (corners*blackAndWhite)
    cnr = (corners > 11) 
    corners = np.zeros(blackAndWhite.shape, np.uint8)
    corners[cnr] = 1
    corners[edgeWidth:-edgeWidth, edgeHeight:-edgeHeight] = 0
    corners = cv2.dilate(corners, np.ones((2,2), np.uint8), iterations=2)

    labeled_array, num_features = scipy.ndimage.label(corners)
    # return (corners, labeled_array, num_features)
    # return labeled_array
    return scipy.ndimage.center_of_mass(corners, labeled_array, np.arange(1,num_features))

def sortCorners(corners):
    arg = np.argsort(corners[:,0])
    corners[:,:] = corners[arg,:]

    tl = corners[1] if corners[0,1] > corners[1,1] else corners[0];
    tr = corners[0] if corners[0,1] > corners[1,1] else corners[1];
    bl = corners[3] if corners[2,1] > corners[3,1] else corners[2];
    br = corners[2] if corners[2,1] > corners[3,1] else corners[3];

    return np.array([tl, bl, br, tr],np.float32)


cap = cv2.VideoCapture(0)

loop = True

while(loop):
    # loop = False

    # Capture frame-by-frame
    ret, frame = cap.read()

    height, width, _ = frame.shape

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    grayCLAHE = clahe.apply(gray)

    thresh,bw = cv2.threshold(grayCLAHE, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2,2), np.uint8)
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw2 = 1-bw2

    labeled_array, num_features = scipy.ndimage.label(bw2, np.ones((3,3),np.uint8))
    areas = scipy.ndimage.find_objects(labeled_array)

    ii = 0
    colourII = 0
    for area in areas:
        rows = area[0].stop - area[0].start
        cols = area[1].stop - area[1].start

        ii += 1

        if rows > 0.6*height:
            continue
        if cols > 0.6*width:
            continue

        if rows < 0.02*height:
            continue
        if cols < 0.02*width:
            continue

        image, contours, heirachy = cv2.findContours(np.copy(bw2[area[0], area[1]]),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if contours[0].shape[0] < 4:
            continue

        epsilon = 0.05*cv2.arcLength(contours[0],True)
        contour = cv2.approxPolyDP(contours[0],epsilon,True)

        if contour.shape[0] != 4:
            continue

        cntArea = cv2.contourArea( contour)
        if cntArea < 64:
            continue

        transW = int(math.sqrt(cntArea)+0.5)

        reducedContour = np.array(contour[:,0,:], np.float32)
        reducedContour = sortCorners(reducedContour)

        cnr = [area[0].start, area[1].start]
        otherCnr = [area[0].start+transW, area[1].start+transW]
        imgSize = grayCLAHE.shape
        if otherCnr[0] >= imgSize[0] or otherCnr[1] >= imgSize[1]:
            continue

        transM = cv2.getPerspectiveTransform(reducedContour, np.array([[0,0], [transW,0], [transW, transW], [0, transW] ], np.float32))
        trans = np.array(cv2.warpPerspective(bw2[area[0], area[1]], transM, (transW, transW) ), np.uint8)

        colour = ((colourII%6)*255/6, 255, 255)
        colourDull = ((colourII%6)*255/6, 82, 255)
        colourII += 1

        hsv[labeled_array == ii ] = colourDull

        contour[:,0,:] = contour[:,0,:] + [area[1].start, area[0].start]
        cv2.drawContours(hsv, [contour], -1, colour, 2)

        grayCLAHE[cnr[0]:otherCnr[0],cnr[1]:otherCnr[1]] = trans*255


    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('res', res)
    # cv2.imshow('frame', frame)
    cv2.imshow('grayCLAHE', grayCLAHE)
    cv2.imshow('bw2', bw2*255)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
