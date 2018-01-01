import numpy as np
import cv2
from matplotlib import pyplot as plt

def nothing(x):
    pass

def findAndDrawContours(cntImg, mask, colour):
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return cv2.drawContours(cntImg, contours, -1, colour, 1)



cv2.namedWindow('control')
cv2.resizeWindow('control', 400, 500)

# create trackbars for color change
cv2.createTrackbar('Filter R','control',3,10,nothing)
cv2.createTrackbar('Board','control',1,1,nothing)
cv2.createTrackbar('Green','control',1,1,nothing)
cv2.createTrackbar('Pink','control',1,1,nothing)
cv2.createTrackbar('Orange','control',1,1,nothing)
cv2.createTrackbar('Yellow','control',1,1,nothing)

cv2.createTrackbar('C1','control',160,255,nothing)
cv2.createTrackbar('C2','control', 90,255,nothing)

cv2.createTrackbar('contains','control', 80,100,nothing)

cap = cv2.VideoCapture(0)

loop = True

while(loop):
    # loop = False

    filterR = cv2.getTrackbarPos('Filter R','control')
    showBoard = cv2.getTrackbarPos('Board','control')
    showGreen = cv2.getTrackbarPos('Green','control')
    showPink = cv2.getTrackbarPos('Pink','control')
    showOrange = cv2.getTrackbarPos('Orange','control')
    showYellow = cv2.getTrackbarPos('Yellow','control')

    CLow = cv2.getTrackbarPos('C1','control')
    CHigh = cv2.getTrackbarPos('C2','control')

    thresh = float(cv2.getTrackbarPos('contains','control'))/100

    # Capture frame-by-frame
    ret, frameRaw = cap.read()
    frame = cv2.bilateralFilter(frameRaw,-1,20,filterR)

    width, height, _ = frameRaw.shape

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    grayCLAHE = clahe.apply(gray)

    kernel = np.ones((filterR,filterR),np.uint8)

    # boardMask  = cv2.morphologyEx(cv2.inRange(hsv, np.array([0,    0, 190]), np.array([255, 135, 255])), cv2.MORPH_OPEN, kernel)
    # greenMask  = cv2.morphologyEx(cv2.inRange(hsv, np.array([25, 130, 130]), np.array([40,  205, 210])), cv2.MORPH_OPEN, kernel)
    # pinkMask   = cv2.morphologyEx(cv2.inRange(hsv, np.array([0,  130, 200]), np.array([10,  195, 255])), cv2.MORPH_OPEN, kernel)
    # orangeMask = cv2.morphologyEx(cv2.inRange(hsv, np.array([10, 190, 200]), np.array([15,  255, 255])), cv2.MORPH_OPEN, kernel)
    # yellowMask = cv2.morphologyEx(cv2.inRange(hsv, np.array([18, 190, 205]), np.array([35,  230, 245])), cv2.MORPH_OPEN, kernel)

    boardMask  = cv2.inRange(hsv, np.array([0,    0, 160]), np.array([255, 135, 255]))
    greenMask  = cv2.inRange(hsv, np.array([20, 100, 130]), np.array([45,  215, 255]))
    pinkMask   = cv2.inRange(hsv, np.array([0,  130, 200]), np.array([10,  195, 255]))
    orangeMask = cv2.inRange(hsv, np.array([10, 190, 200]), np.array([15,  255, 255]))
    yellowMask = cv2.inRange(hsv, np.array([18, 190, 205]), np.array([35,  230, 245]))

    edges = cv2.Canny(grayCLAHE,CLow,CHigh, filterR)

    kernel = np.ones((3,3),np.uint8)
    edges  = cv2.dilate(edges, kernel)
    kernel = np.ones((2,2),np.uint8)
    edges  = cv2.erode(edges, kernel)
    edges  = cv2.erode(edges, kernel)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    masks = [[ 'white',  boardMask,  (255,255,255), showBoard], 
             [ 'green',  greenMask,  (000,255,000), showGreen], 
             [ 'pink',   pinkMask,   (200,000,255), showPink],
             [ 'orange', orangeMask, (000,180,255), showOrange],
             [ 'yellow', yellowMask, (000,255,255), showYellow]]
    
    cntImg = np.zeros((width, height, 3),np.uint8)
    mask = np.zeros((width, height),np.uint8)

    areas = []
    kernel = np.ones((3,3),np.uint8)

    for i in range( 0, len( contours ) ):
        epsilon = 0.005*cv2.arcLength(contours[i],True)
        contours[i] = cv2.approxPolyDP(contours[i],epsilon,True)

        zeros = np.zeros((width, height),np.uint8)
        contourMask = cv2.drawContours( zeros, contours, i, 1, -1)
        contourMask = cv2.morphologyEx(contourMask, cv2.MORPH_CLOSE, kernel)
        contourMaskSum = np.sum(contourMask)
        if contourMaskSum > 10:
            areas += [ [ contourMaskSum, contourMask] ]

    for area in areas:
        contourMaskSum = area[0]
        contourMask = area[1]
        for colourMask in masks:
            if colourMask[3]:
                mask = cv2.add(mask, colourMask[1])
            res = cv2.bitwise_and(contourMask, colourMask[1])
            res = cv2.dilate(res, kernel)
            if  np.sum( res ) > (thresh*contourMaskSum):
                cntImg[contourMask==1] = colourMask[2]
                # cntImg = cv2.drawContours( cntImg, [contourMask], -1, colourMask[2], -1 )
                # break


    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    cv2.imshow('grayCLAHE',grayCLAHE)

    cv2.imshow('contours',cntImg)
    cv2.imshow('edges',edges)

    cv2.imshow('res',res)

    # for i in range(0,3):
    #     histr = cv2.calcHist([hsv],[i],None,[256],[0,256])
    #     cdf = histr.cumsum()
    #     cdf_n = cdf* histr.max()/cdf.max()
    #     cdf_m = np.ma.masked_equal(cdf,0)
    #     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    #     cdf = np.ma.filled(cdf_m,0).astype('uint8')
    #     hsv[:,:,i] = cdf[hsv[:,:,i]]

    # hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('norm',hsv_bgr)

    # color = ('b','g','r')
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([hsv],[i],None,[256],[0,256])
    #     cdf = histr.cumsum()
    #     cdf_n = cdf* histr.max()/cdf.max()
    #     cdf_m = np.ma.masked_equal(cdf,0)
    #     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    #     cdf = np.ma.filled(cdf_m,0).astype('uint8')

    #     plt.plot(cdf_n,color = 'k')
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # plt.show()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
