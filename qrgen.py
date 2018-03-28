import numpy as np
import cv2
from code import *

bw = blockWidth
scale = 5
sd =np.arange(0,blockWidth*20)*scale

img = generateImage(42, 4,3,scale)
showDontWeight(img/255, '42')

im = img[sd[1]:sd[bw-1],sd[1]:sd[bw-1]]/255

values = extractInner(im)
print('final:',values[0])


# showDontWeight(values[1], 'inner')

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# for ii in range(0,128):
#     img = generateImage(ii, 5,4,5)
#     cv2.imwrite('img/'+str(ii)+'.png',img)


