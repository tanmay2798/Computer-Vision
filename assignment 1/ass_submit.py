import cv2 
import numpy as np
import array as arr

  
# Function to extract frames 
def FrameCapture(path,name): 


  	cap = cv2.VideoCapture(path)
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(name, fourcc, fps, (width,height))
	x1=[]
	y1=[]
	x2=[]
	y2=[]
	avgx1=0
	avgx2=0
	avgy1=0
	avgy2=0
	fx=0
	fy=0
	m=0
	x_min_line=0
	y_min_line=0
	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		kernel = np.ones((5,5),np.uint8)
		#erosion = cv2.erode(fgmask,kernel,iterations = 1)
		#dilation = cv2.dilate(erosion,kernel,iterations = 1)
		#gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
		blur = cv2.GaussianBlur(fgmask, (7, 7), 0)
		closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
		grad_x = cv2.Sobel(closing, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		grad_y = cv2.Sobel(closing, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		abs_grad_x = cv2.convertScaleAbs(grad_x)
		abs_grad_y = cv2.convertScaleAbs(grad_y)
		grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
		lines = cv2.HoughLinesP(grad,1,np.pi/180,280, maxLineGap = 10)
		#print lines

		if lines is not None:
			x1=[]
			y1=[]
			x2=[]
			y2=[]
			for l in lines:
				x1.append(l[0][0])
				y1.append(l[0][1])
				x2.append(l[0][2])
				y2.append(l[0][3])
				#frame = cv2.line(frame,(l[0][0],l[0][1]),(l[0][2],l[0][3]),(0,255,0),2)
			avgx1=sum(x1)/len(x1)
			avgx2=sum(x2)/len(x2)
			avgy1=sum(y1)/len(y1)
			avgy2=sum(y2)/len(y2)
			if avgx1 != avgx2 and avgy1!=avgy2:
				m = (avgy2-avgy1)/(avgx2-avgx1)
				fx=(avgx1+avgx2)/2
				fy=(avgy1+avgy2)/2
			
			minimum = y1.index(min(y1))
			x_min = max(x2)
			y_min = max(y2)
			x_min_line = (x_min + m*y_min + m*m*avgx2 - m*avgy2)/(m*m+1)
			y_min_line = (avgy2+m*x_min_line - m*avgx2)

			if (fy-1000*m<fy+1000*m):
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx-1000,fy-1000*m),(255,0,0),6)
			else:
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx+1000,fy+1000*m),(255,0,0),6)
			
		else:	
			if (fy-1000*m<fy+1000*m):
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx-1000,fy-1000*m),(255,0,0),6)
			else:
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx+1000,fy+1000*m),(255,0,0),6)

		k = cv2.waitKey(30) & 0xff
		if k == 27:
		    break
		cv2.imshow('edges',fgmask)
		cv2.imshow('frame',frame)
		out.write(frame)
	cap.release()
	out.release()

  
# Driver Code 
if __name__ == '__main__': 

    try:
    	FrameCapture("1.mp4","1.avi")
    except:
    	print 'l'
	try:
		FrameCapture("2.mp4","2.avi")
	except:
		print 'l' 
	try:
		FrameCapture("3.mp4","3.avi")
	except:
		print 'l' 
	try:
		FrameCapture("4.mp4","4.avi")
	except:
		print 'l' 
	try:
		FrameCapture("5.mp4","5.avi")
	except:
		print 'l' 
	try:
		FrameCapture("6.mp4","6.avi")
	except:
		print 'l' 
	try:
		FrameCapture("7.mp4","7.avi")
	except:
		print 'l' 
	try:
		FrameCapture("8.mp4","8.avi")
	except:
		print 'l'  
	try:
		FrameCapture("9.mp4","9.avi")
	except:
		print 'l' 
	try:
		FrameCapture("10.mp4","10.avi")
	except:
		print 'l'


