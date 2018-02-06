import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from skimage.filters import threshold_local

image = cv2.imread('receipt.jpg')
ratio = image.shape[0]/500
orig = image.copy()
image = imutils.resize(image,height = 500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blur,75,200)
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]
cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:5]
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
pts = screenCnt.reshape(4,2)*ratio
rect = np.zeros((4,2),'float32')
pts = np.array(eval('pts'), dtype = "float32")
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]
diff = np.diff(pts,axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
(tl,tr,br,bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect,dst)
wrap = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
wrap = cv2.cvtColor(wrap,cv2.COLOR_BGR2GRAY)
T = threshold_local(wrap, 11, offset = 10, method = "gaussian")
wrap = (wrap>T).astype("uint8")*255
cv2.imshow('Scanned',wrap)
cv2.waitKey(0)
cv2.destroyAllWindows()
