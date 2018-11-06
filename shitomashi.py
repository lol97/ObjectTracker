import cv2
import numpy as np
from matplotlib import pyplot as plt

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
kernel = np.array([[1,0,1],[0,1,0],[1,0,1]])
video = cv2.VideoCapture("bola.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
while (video.isOpened()):
	ret, frame = video.read()
	# plt.close()
	if(int(video.get(1))%2==0):
		# print(int(video.get(1)))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		gray = cv2.dilate(th3,kernel,iterations=1)
		gray = cv2.dilate(th3,kernel,iterations=1)
		gray = cv2.dilate(th3,kernel,iterations=1)
		corners = cv2.goodFeaturesToTrack(gray,mask=None, **feature_params)
		crn = np.int0(corners)
		print(len(crn))
		for x in crn:
			post = tuple(x[0].tolist())
			cv2.circle(frame,post,1,(255,0,0),1)
		# print(corners)
		cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()