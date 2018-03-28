import numpy as np
import scipy.ndimage
import cv2
from matplotlib import pyplot as plt
import math
import code 

def nothing(x):
    pass

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
    # frame = cv2.blur(frame, (3,3))

    height, width, _ = frame.shape

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11,11))
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

        epsilon = 0.03*cv2.arcLength(contours[0],True)
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
        transGray = np.array(cv2.warpPerspective(grayCLAHE[area[0], area[1]], transM, (transW, transW) ), np.uint8)

        thresh,trans = cv2.threshold(transGray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # trans = np.array(cv2.warpPerspective(bw2[area[0], area[1]], transM, (transW, transW) ), np.uint8)
        # trans = 1-trans

        # kernel = np.ones((3,3), np.uint8)
        # trans = cv2.morphologyEx(trans, cv2.MORPH_OPEN, kernel, iterations=1)
        # trans = cv2.morphologyEx(trans, cv2.MORPH_CLOSE, kernel, iterations=1)
        # kernel[1,0] = 0
        # kernel[0,1] = 0
        # kernel[1,-1] = 0
        # kernel[-1,1] = 0
        # trans = cv2.morphologyEx(trans, cv2.MORPH_OPEN, kernel, iterations=1)
        # kernel = np.ones((3,3), np.uint8)
        # trans = cv2.morphologyEx(trans, cv2.MORPH_OPEN, kernel, iterations=1)
        # kernel = np.ones((3,3), np.uint8)
        # trans = cv2.morphologyEx(trans, cv2.MORPH_CLOSE, kernel, iterations=1)
        # kernel[0,0] = 0
        # kernel[0,-1] = 0
        # kernel[-1,-1] = 0
        # kernel[-1,0] = 0
        # trans = cv2.morphologyEx(trans, cv2.MORPH_CLOSE, kernel, iterations=1)

        # trans = 1-trans

        values = code.extractInner(trans)
        # if values[0] == -1 :
        #     continue
        # data = code.decodeInner(values[1], values[2])

        colour = ((colourII%6)*255/6, 255, 255)
        colourDull = ((colourII%6)*255/6, 82, 255)
        colourII += 1

        hsv[labeled_array == ii ] = colourDull

        contour[:,0,:] = contour[:,0,:] + [area[1].start, area[0].start]
        cv2.drawContours(hsv, [contour], -1, colour, 2)


        grayCLAHE[cnr[0]:otherCnr[0],cnr[1]:otherCnr[1]] = trans*255
        # grayCLAHE[cnr[0]:otherCnr[0],cnr[1]:otherCnr[1]] = transGray

        if values[0] != -1:
            data = code.decodeInner(values[1], values[2])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(grayCLAHE,str(data[0])+"("+str(data[1])+", "+str(data[2])+") "+str(values[0])
                    ,(cnr[1],cnr[0]), font, 0.4,(255),1,cv2.LINE_AA)


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
