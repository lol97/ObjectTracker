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

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

#init awal
ret, frame = video.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(old_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
old_gray = cv2.dilate(th3,kernel,iterations=1)
# old_gray = cv2.dilate(th3,kernel,iterations=1)
# old_gray = cv2.dilate(th3,kernel,iterations=1)
p0 = cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
mask = np.zeros_like(frame)


while (video.isOpened()):
	ret, frame = video.read()
	# plt.close()
	if(int(video.get(1))%4==0):
		# print(int(video.get(1)))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		gray = cv2.dilate(th3,kernel,iterations=1)
		# gray = cv2.dilate(th3,kernel,iterations=1)
		# gray = cv2.dilate(th3,kernel,iterations=1)
		corners = cv2.goodFeaturesToTrack(gray,mask=None, **feature_params)
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
		# print(type(p1))
		# Select good points
		print(p1 is not None)
		if(p1 is not None):
			good_new = p1[st==1]
			good_old = p0[st==1]

			# draw the tracks
			for i,(new,old) in enumerate(zip(good_new,good_old)):
				a,b = new.ravel()
				c,d = old.ravel()
				mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
				Nframe = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
				img = cv2.add(Nframe,mask)

			cv2.imshow("track",Nframe)
			crn = np.int0(corners)
			print(len(crn))
			for x in crn:
				post = tuple(x[0].tolist())
				cv2.circle(frame,post,1,(255,0,0),1)
			# print(corners)
			cv2.imshow('frame',frame)
			old_gray = gray.copy()
			p0 = good_new.reshape(-1,1,2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()