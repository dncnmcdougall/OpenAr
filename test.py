import numpy as np
import scipy.ndimage
import cv2
from matplotlib import pyplot as plt

def nothing(x):
    pass

def findCorners(bw):
    width, height = bw.shape
    innerImg = np.ones((width+6, height+6), np.uint8)
    innerImg[3:-3,3:-3] = bw
    corners = np.ones(bw.shape,np.uint8)*20
    for ii in range(1,6):
        corners -= innerImg[0:-6,ii:(ii-6)]
        corners -= innerImg[6: , ii:(ii-6)]
        corners -= innerImg[ii:(ii-6),0:-6]
        corners -= innerImg[ii:(ii-6),6:  ]
    corners = (corners*bw)
    cnr = (corners > 11) 
    corners = np.zeros(bw.shape, np.uint8)
    corners[cnr] = 1
    return corners

cap = cv2.VideoCapture(0)

loop = True

while(loop):
    # loop = False

    # Capture frame-by-frame
    ret, frame = cap.read()

    width, height, _ = frame.shape

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    grayCLAHE = clahe.apply(gray)

    thresh,bw = cv2.threshold(grayCLAHE, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    kernel = np.ones((2,2), np.uint8)
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw2 = 1-bw2

    corners = findCorners(bw2)

    labeled_array, num_features = scipy.ndimage.label(bw2)
    print(num_features)

    hsv[:,:,0] = (labeled_array%6)*(255/6) + hsv[:,:,0]*(1-bw2)
    hsv[:,:,1] = 255*(bw2) + hsv[:,:,1]*(1-bw2)
    hsv[:,:,2] = 255*(bw2) + hsv[:,:,2]*(1-bw2)

    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('res', res)
    cv2.imshow('frame', frame)
    cv2.imshow('grayCLAHE', grayCLAHE)
    cv2.imshow('bw', bw*255)
    cv2.imshow('bw2', bw2*255)
    cv2.imshow('corners', corners*255)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
