import cv2

from keypoint_data import KeypointData

img = cv2.imread('car.png')
img_mask = cv2.imread('car_mask.png', 0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.05, edgeThreshold=30, sigma=1.5)
kp, des = sift.detectAndCompute(gray,img_mask)

img = cv2.drawKeypoints(img, keypoints = kp)
cv2.imwrite('sift_keypoints.png', img)

k = KeypointData(kp, des)
k.save("keypoints.p")
