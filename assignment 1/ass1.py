# Program To Read video 
# and Extract Frames 
import cv2 
import numpy as np
import array as arr

  
# Function to extract frames 
def FrameCapture(path,name): 
        # vidObj object calls read 
        # function extract frames 

  	cap = cv2.VideoCapture(path)
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(name, fourcc, fps, (width,height))
	#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
	'''x1max=arr.array('i')
	x1min=arr.array('i')
	x2max=arr.array('i')
	x2min=arr.array('i')
	y1max=arr.array('i')
	y1min=arr.array('i')
	y2max=arr.array('i')
	y2min=arr.array('i')'''
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
		edges = cv2.Canny(grad,50,150,apertureSize = 3)
		lines = cv2.HoughLinesP(grad,1,np.pi/180,280, maxLineGap = 10)
		#print lines

		if lines is not None:
			'''del x1max[:]
			del x2max[:]
			del y1max[:]
			del y2max[:]
			del x1min[:]
			del x2min[:]
			del y1min[:]
			del y2min[:]'''
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
			#avgy=(max(y1,y2)+min(y1,y2))/2
		#	frame = cv2.line(frame,((max(x1)+min(x1))/2,(max(y1)+min(y1))/2),((max(x2)+min(x2))/2,(max(y2)+min(y2))/2),(0,255,0),5)
			#frame = cv2.line(frame,(avgx1+100*slope,avgy1+100*slope),(avgx2-100*slope,avgy2-100*slope),(0,255,0),5)
			
			#frame = cv2.line(frame,(avgx1,avgy1),(avgx2,avgy2),(255,0,0),4)
			if (fy-1000*m<fy+1000*m):
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx-1000,fy-1000*m),(255,0,0),6)
			else:
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx+1000,fy+1000*m),(255,0,0),6)
			
			#print m
			#frame = cv2.line(frame,(avgx1+500,m*500+avgy1),(avgx2-1000,avgy2-1000*m),(0,255,0),5)
			#frame = cv2.line(frame,(fx+1000,m*1000+fy),(fx-1000,fy-1000*m),(255,0,0),5)
		else:	
			if (fy-1000*m<fy+1000*m):
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx-1000,fy-1000*m),(255,0,0),6)
			else:
				frame = cv2.line(frame,(int(x_min_line),int(y_min_line)),(fx+1000,fy+1000*m),(255,0,0),6)
			#frame = cv2.line(frame,(fx+1000,m*1000+fy),(fx-1000,fy-1000*m),(255,0,0),5)
		#	print 'ji'
			#frame = cv2.line(frame,(avgx1+500,m*500+avgy1),(avgx1-1000,avgy1-1000*m),(0,255,0),5)
			#frame = cv2.line(frame,(avgx1,avgy1),(avgx2,avgy2),(0,255,0),2)
			
			
			#frame = cv2.rectangle(frame,(min(x1),min(y1)),(max(x2),max(y2)),(0,255,0),5)
		'''else:
			avgx1=sum(x1)/len(x1)
			avgx2=sum(x2)/len(x2)
			avgy1=sum(y1)/len(y1)
			avgy2=sum(y2)/len(y2)
			frame = cv2.line(frame,(avgx1,avgy1),(avgx2,avgy2),(0,255,0),5)'''

			#cv2.imshow('frame',edges)
		#cv2.imshow('frame2',gradient)
		#print 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
		k = cv2.waitKey(30) & 0xff
		if k == 27:
		    break
		cv2.imshow('edges',fgmask)
		cv2.imshow('frame',frame)
		out.write(frame)
		'''hwnd = winGuiAuto.findTopWindow("frame")
		win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
		rect = win32gui.GetWindowPlacement(hwnd)[-1]
		image = ImageGrab.grab(rect)
		image_list.append(image)
		#cv2.imshow('g',fgmask)'''
		
		#break
	cap.release()
	out.release()
	'''out = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, (width,height))
 
	for images in image_list:
	    out.write(cv2.cvtColor(np.array(images),cv2.COLOR_BGR2RGB))
	out.release()'''
	#cv2.destroyAllWindows()

  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function  

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


