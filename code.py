import numpy as np
import scipy.misc
import cv2

# size of a code
blockWidth=9
idWidth = 7
sideWidth = 4
# blockWidth= 10
# idWidth = 10
# sideWidth = 5

# decoding thresholds
threshold = 0.045 #direction threshold
boxThreshold = 0.4 # 10 threshold

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

    return img


# these are convenience methods for debugging
def showDontWait(im, name='show'):
    img = im*255
    cv2.imshow(name,img)

def showAndWait(im, name='show'):
    img = im*255
    cv2.imshow(name,img)
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def extractInnerNoRotationNoScaling(img, scale, mask):
    bw=blockWidth-2
    sd =np.arange(0,bw+1)*scale

    thresh = threshold*scale*scale*bw*bw

    def addToOutput(outside, img, rows, cols):
        outside[rows,cols] = img[rows,cols]

    outside = np.ones((bw*scale,bw*scale), np.uint8)
    addToOutput(outside, img, slice(0,sd[2]),         slice(0,sd[bw]))
    addToOutput(outside, img, slice(sd[bw-1],sd[bw]), slice(0,sd[bw])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(0,sd[2])) 
    addToOutput(outside, img, slice(0,sd[bw]), slice(sd[bw-1],sd[bw]))

    # This expands the inner bit a little. 
    # The idea is that this avoids dodgy overlays, and slight rotations, thus improving robustness.
    extraBit = scale/5
    innerSlice = slice((sd[2]-extraBit),(sd[bw-1] + extraBit))
    outside[innerSlice, innerSlice] = 1
    mask[innerSlice, innerSlice] = 1

    result = sum(sum(mask != outside))

    if result < thresh:
        inner = img[sd[2]:sd[bw-1], sd[2]:sd[bw-1]]
        return (inner, result)
    else:
        return (None, -1)

# Takes in a square image without the white border
def extractInner(im):
    side, cols = im.shape
    assert(side == cols)
    bw=blockWidth-2

    scale = side/bw
    sd =np.arange(0,bw+1)*scale

    # make the image the same size as the mask
    img = scipy.misc.imresize(im, (bw*scale, bw*scale), 'nearest')

    # construct the direction mask
    mask = np.ones((bw*scale,bw*scale), np.uint8)
    # top and bottom borders
    mask[0:sd[1],       0:sd[bw]] = 0
    mask[sd[bw-1]:sd[bw], 0:sd[bw]] = 0
    # left and right borders
    mask[0:sd[bw], 0:sd[1]] = 0
    mask[0:sd[bw], sd[bw-1]:sd[bw]] = 0
    # timing dots
    for ii in range(1,bw-1,2):
        jj = ii
        mask[sd[jj]:sd[jj+1], sd[1]:sd[2]] = 0
        mask[sd[1]:sd[2], sd[jj]:sd[jj+1]] = 0

    # extract all four directions
    values = []
    rotImg =  img
    result = extractInnerNoRotationNoScaling(rotImg, scale, mask)
    if result[1] >= 0:
        values += [[result[1], 1, result[0]]]

    rotImg = np.transpose(img)[:,-1::-1]
    result = extractInnerNoRotationNoScaling(rotImg, scale, mask)
    if result[1] >= 0:
        values += [[result[1], 2, result[0]]]

    rotImg =  img[-1::-1,-1::-1]
    result = extractInnerNoRotationNoScaling(rotImg, scale, mask)
    if result[1] > 0:
        values += [[result[1], 3, result[0]]]

    rotImg =  np.transpose(img)[-1::-1,:]
    result = extractInnerNoRotationNoScaling(rotImg, scale, mask)
    if result[1] >= 0:
        values += [[result[1], 4, result[0]]]

    # if none of the directions is a valid candidate, retune none
    if len(values) == 0:
        return (-1, None, scale)

    # return the best candidate
    minVal = values[0][0]
    for ii in range(1, len(values)):
        minVal = min(values[ii][0], minVal)

    ii = [ii for ii in range(0, len(values)) if values[ii][0] == minVal][0]
    return (values[ii][1], values[ii][2], scale)

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


