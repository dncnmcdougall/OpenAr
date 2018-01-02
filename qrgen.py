import numpy as np
import cv2

def generateImage(id, width, height, scale):

    blockWidth=10
    bw=blockWidth

    dataWidth=blockWidth-5
    totalValue= (width<<15) + (height<<10) + id

    # scaled distance
    sd =np.arange(0,blockWidth*max(width,height))*scale

    pixelsWide = width*scale*blockWidth
    pixelsHigh = height*scale*blockWidth
    img = np.ones((pixelsHigh, pixelsWide),np.uint8)*255

    mask = np.ones((bw*scale,bw*scale), np.uint8)*255
    mask[sd[1]:sd[2],       sd[1]:sd[bw-1]] = 0
    mask[sd[bw-2]:sd[bw-1], sd[1]:sd[bw-1]] = 0

    mask[sd[1]:sd[bw-1], sd[1]:sd[2]] = 0
    mask[sd[1]:sd[bw-1], sd[bw-2]:sd[bw-1]] = 0

    for ii in range(1,bw-1,2):
        jj = ii+1
        mask[sd[jj]:sd[jj+1], sd[2]:sd[3]] = 0
        mask[sd[2]:sd[3], sd[jj]:sd[jj+1]] = 0

    for ii in range(0,dataWidth*dataWidth-1):
        column = (ii%dataWidth) +3
        row = ii/dataWidth +3
        value = (1-((totalValue >> ii) & 1))*255
        mask[sd[row]:sd[row+1], sd[column]:sd[column+1]] = value

    img[0:bw*scale,0:bw*scale] = mask
    img[pixelsHigh:pixelsHigh-(bw*scale)-1:-1, pixelsWide:pixelsWide-(bw*scale)-1:-1] = mask
    img[0:bw*scale, pixelsWide:pixelsWide-(bw*scale)-1:-1] = mask
    img[pixelsHigh:pixelsHigh-(bw*scale)-1:-1, 0:bw*scale] = mask


    return img


img = generateImage(42, 4,3,5)
cv2.imwrite('42.png',img)
cv2.imshow('42',img)
img = generateImage(49, 4,3,5)
cv2.imwrite('49.png',img)
cv2.imshow('49',img)
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

