import numpy as np
import scipy.misc

blockWidth=9
idWidth = 7
sideWidth = 4
# blockWidth= 10
# idWidth = 10
# sideWidth = 5

threshold = 0.045
boxThreshold = 0.4

dataWidth=blockWidth-5
if (dataWidth*dataWidth-1) < ( 2*sideWidth + idWidth):
    print( (dataWidth*dataWidth-1) , ( 2*sideWidth + idWidth))
    assert((dataWidth*dataWidth-1) >= ( 2*sideWidth + idWidth))


def generateImage(id, width, height, scale):
    bw=blockWidth

    assert( id < 2**idWidth)
    assert( width < 2**sideWidth)
    assert( height < 2**sideWidth)

    totalValue= (width<<(sideWidth + idWidth)) + (height<<idWidth) + id

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

    for ii in range(1,bw-2,2):
        jj = ii+1
        mask[sd[jj]:sd[jj+1], sd[2]:sd[3]] = 0
        mask[sd[2]:sd[3], sd[jj]:sd[jj+1]] = 0

    for ii in range(0,dataWidth*dataWidth-1):
        column = (ii%dataWidth) +3
        row = ii/dataWidth +3
        value = (1-((totalValue >> ii) & 1))*255
        mask[sd[row]:sd[row+1], sd[column]:sd[column+1]] = value

    img[0:bw*scale,0:bw*scale] = mask
    img[(-bw*scale-1):-1,0:bw*scale] = np.transpose(mask)[-1::-1,:]
    img[(-bw*scale-1):-1,(-bw*scale-1):-1] = mask[-1::-1,-1::-1]
    img[0:bw*scale,(-bw*scale-1):-1] = np.transpose(mask)[:,-1::-1]

    # img[0:bw*scale,0:bw*scale] = mask
    # img[-1:-bw*scale-1:-1,0:bw*scale] = np.transpose(mask)
    # img[-1:-bw*scale-1:-1,-1:-bw*scale-1:-1] = mask
    # img[0:bw*scale,-1:-bw*scale-1:-1] = np.transpose(mask)

    return img

import cv2

def showAndWeight(im, name='show'):
    img = im*255
    cv2.imshow(name,img)
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Takes in a square image without the white border
def extractInner(im):
    side, cols = im.shape
    assert(side == cols)
    bw=blockWidth-2

    scale = side/bw
    sd =np.arange(0,bw+1)*scale

    img = scipy.misc.imresize(im, (bw*scale, bw*scale), 'nearest')

    mask = np.ones((bw*scale,bw*scale), np.uint8)

    mask[0:sd[1],       0:sd[bw]] = 0
    mask[sd[bw-1]:sd[bw], 0:sd[bw]] = 0

    mask[0:sd[bw], 0:sd[1]] = 0
    mask[0:sd[bw], sd[bw-1]:sd[bw]] = 0

    for ii in range(1,bw-1,2):
        jj = ii
        mask[sd[jj]:sd[jj+1], sd[1]:sd[2]] = 0
        mask[sd[1]:sd[2], sd[jj]:sd[jj+1]] = 0

    thresh = threshold*scale*scale*bw*bw

    def addToOutput(outside, img, rows, cols):
        outside[rows,cols] = img[rows,cols]

    outside = np.ones((bw*scale,bw*scale), np.uint8)
    addToOutput(outside, img, slice(0,sd[2]),         slice(0,sd[bw]))
    addToOutput(outside, img, slice(sd[bw-1],sd[bw]), slice(0,sd[bw])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(0,sd[2])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(sd[bw-1],sd[bw]))
    inner = img[sd[2]:sd[bw-1], sd[2]:sd[bw-1]]


    rotMask =  mask

    result = sum(sum(rotMask != outside))
    if result < thresh:
        return (1, inner, scale)

    outside = np.ones((bw*scale,bw*scale), np.uint8)
    addToOutput(outside, img, slice(0,sd[1]),         slice(0,sd[bw]))
    addToOutput(outside, img, slice(sd[bw-2],sd[bw]), slice(0,sd[bw])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(0,sd[2])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(sd[bw-1],sd[bw]))
    inner = img[sd[1]:sd[bw-2], sd[2]:sd[bw-1]]

    rotMask =  np.transpose(mask)[-1::-1,:]

    result = sum(sum(rotMask != outside))
    if result < thresh:
        return (2, np.transpose(inner)[:,-1::-1], scale)

    outside = np.ones((bw*scale,bw*scale), np.uint8)
    addToOutput(outside, img, slice(0,sd[1]),         slice(0,sd[bw]))
    addToOutput(outside, img, slice(sd[bw-2],sd[bw]), slice(0,sd[bw])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(0,sd[1])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(sd[bw-2],sd[bw]))
    inner = img[sd[1]:sd[bw-2], sd[1]:sd[bw-2]]

    rotMask =  mask[-1::-1,-1::-1]

    result = sum(sum(rotMask != outside))
    if result < thresh:
        return (3, inner[-1::-1,-1::-1], scale)

    outside = np.ones((bw*scale,bw*scale), np.uint8)
    addToOutput(outside, img, slice(0,sd[2]),         slice(0,sd[bw]))
    addToOutput(outside, img, slice(sd[bw-1],sd[bw]), slice(0,sd[bw])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(0,sd[1])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(sd[bw-2],sd[bw]))
    inner = img[sd[2]:sd[bw-1], sd[1]:sd[bw-2]]

    rotMask = np.transpose(mask)[:,-1::-1]

    result = sum(sum(rotMask != outside))
    if result < thresh:
        return (4, np.transpose(inner)[-1::-1,:], scale)

    return (-1, None, scale)

def decodeInner(img, scale):

    assert( img.shape[0]/scale == dataWidth )

    sd =np.arange(0,dataWidth+1)*scale

    totalValue = 0
    for ii in range(0,dataWidth*dataWidth-1):
        col = (ii%dataWidth) 
        row = ii/dataWidth
        data = (sum(sum(img[sd[row]:sd[row+1],sd[col]:sd[col+1]])) < boxThreshold*(scale*scale))
        totalValue += data << ii

    width = totalValue >> (idWidth+sideWidth)
    totalValue -= width << (idWidth+sideWidth)

    height = totalValue >> (idWidth)
    totalValue -= height << (idWidth)

    id = totalValue

    return (id, width, height)


