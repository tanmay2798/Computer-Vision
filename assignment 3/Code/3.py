import argparse

import cv2
import numpy as np
import time
import math
import os
import copy
from objloader_simple import *

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    
def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) *(0.8)
    h, w = model.shape
    img1=copy.deepcopy(img)
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        #points = np.dot(points,[[0,-1,0],[1,0,0],[0,0,1]])
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
       # img=cv2.circle(img,(p[0],p[1]), 15, (0,0,255), -1)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        #print(imgpts)
       # img=cv2.circle(img,(imgpts[0][0][0],imgpts[0][0][1]), 15, (0,0,255), -1)
        X=imgpts[0][0][0]
        Y=imgpts[0][0][1]
        if color is False:
            cv2.fillConvexPoly(img1, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img1, imgpts, color)
        
    cv2.imshow('frame1', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

    return img1,X,Y
    
def renderAtPoint(img, obj, projection, model,  x, y, color=False):
    vertices = obj.vertices
    scale = 0.3
    scale_matrix = np.eye(3) * scale
    h, w = model.shape
    img1=copy.deepcopy(img)
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w/2 , p[1] + h/2 , p[2]] for p in points])
       # img=cv2.circle(img,(p[0],p[1]), 15, (0,0,255), -1)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        for cnt in imgpts:
        	for p in cnt:
        		p[0] = p[0] + x
        		p[1] = p[1] + y
        #print(imgpts)
       # img=cv2.circle(img,(imgpts[0][0][0],imgpts[0][0][1]), 15, (0,0,255), -1)
        X=imgpts[0][0][0]
        Y=imgpts[0][0][1]
        if color is False:
            cv2.fillConvexPoly(img1, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img1, imgpts, color)
        
    cv2.imshow('frame1', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

    return img1,X,Y


MIN_MATCHES = 10
X=0
Y=0

def main():


    dir_name = os.getcwd() 
    homography = None
    
	#-------------- CAMERA MATRIX ---------------
    camera_parameters = np.array([[601.1,0,332.86],[0,600.21,226.4],[0,0,1]])
	
	#-------------- IMAGES ---------------
    #model = cv2.imread('two.jpg', 0)
    model = cv2.imread('3.png', 0)
    #frame = cv2.imread('test4.jpg')
    obj = OBJ('models/cow.obj', swapyz=True)	
    
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    fps = int(8)	
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('22.avi', fourcc, fps, (width,height))
    kp_model, des_model = sift.detectAndCompute(model, None)
    
    ret = True
    while(ret):
		ret, frame = cap.read()
		kp_frame, des_frame = sift.detectAndCompute(frame, None)
		
		matches = flann.knnMatch(des_model,des_frame,k=2)
		refined_matches=[]
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				refined_matches.append(m)
		
		
		homography1 = None
		if len(refined_matches) > MIN_MATCHES:
			src_pts = np.float32([ kp_model[m.queryIdx].pt for m in refined_matches ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in refined_matches ]).reshape(-1,1,2)
			homography1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			if homography1 is not None:
				h, w = model.shape
				pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
				dst = cv2.perspectiveTransform(pts, homography1)
				frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 

		else:
		    print "Not enough matches found in 1- %d/%d" % (len(refined_matches), MIN_MATCHES)
		    
		print 1
		if homography1 is not None:
			#tm = np.array([[1, 0, i*tx/steps],[0, 1, i*ty/steps],[0, 0 ,1]])
			#homography1 = np.dot(tm, homography1)
			projection = projection_matrix(camera_parameters, homography1)
			frame,src_X,src_Y = render(frame, obj, projection, model, False)
		
		out.write(frame)
		'''cv2.imshow('window',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break'''
		        
if __name__ == '__main__':
    main()