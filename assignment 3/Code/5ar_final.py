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
    
def get_perpendicular(camera_parameters, homography):
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
    results=[]
    [results.append(y) for y in rot_3]
    #results.pop()
    #print(np.array(results).reshape(-1,1,3),'ll',homography)
    #rot_3[0] = rot_3[0]
    #rot_3[1] = rot_3[0][1]
    #rot_3[1] = [0,1,0]
    #rot_3[2] = [0,0,1]
    #rot_3 = np.array([[results[0],results[1],results[2]],[0,1,0],[0,0,1]])
    #print(rot_3.reshape(-1,1,3),'rot3',homography)
    #ret = cv2.perspectiveTransform(rot_3.reshape(-1,1,3), homography)
    return rot_3

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    
def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.2
    h, w = model.shape
    img1=copy.deepcopy(img)
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
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
    
def check_distance(c, d, cnt):
    ret = False
    dist = cv2.pointPolygonTest(cnt,tuple(c),True)
    print(dist)
    if (dist > 0): ret = True
    return ret
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def changedir(d, camera_matrix, homography,arr,c):
    print('kk',arr[0])
    perp = get_perpendicular(camera_matrix,homography)
    #print('ll',perp)
    perp = [arr[0][0]-c[0],arr[0][1]-c[1]]
    normal = normalize(perp[:2])
    d_norm = normalize(d)
    theta = np.dot(d_norm, normal)
    r = (d_norm - normal*(theta*2))
    return 5*r

MIN_MATCHES = 10
X=0
Y=0

def main():


    dir_name = os.getcwd() 
    homography = None
    
    #-------------- CAMERA MATRIX ---------------
    camera_parameters = np.array([[601.1,0,332.86],[0,600.21,226.4],[0,0,1]])
    
    #-------------- IMAGES ---------------
    model1 = cv2.imread('one.jpg', 0)
    model2 = cv2.imread('two.jpg', 0)
    #frame = cv2.imread('test4.jpg')
    obj = OBJ('models/rat.obj', swapyz=True)    
    
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    fps = int(8)    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('24.avi', fourcc, fps, (width,height))
    kp_model1, des_model1 = sift.detectAndCompute(model1, None)
    kp_model2, des_model2 = sift.detectAndCompute(model2, None)
    c = [ width/10, 2.2*height]
    d = [10, 0]
    
    ret = True
    check = False
    while(ret):
        #ret, frame = cap.read()
        frame=cv2.imread('8.png')
        kp_frame, des_frame = sift.detectAndCompute(frame, None)
        #find matches for 1st marker
        matches1 = flann.knnMatch(des_model1,des_frame,k=2)
        refined_matches1=[]
        for m,n in matches1:
            if m.distance < 0.7*n.distance:
                refined_matches1.append(m)
        #find matches for 2nd marker
        matches2 = flann.knnMatch(des_model2,des_frame,k=2)
        refined_matches2=[]
        for m,n in matches2:
            if m.distance < 0.7*n.distance:
                refined_matches2.append(m)
        
        #homography
        homography1 = None
        hc1 = None
        hc2 = None
        if len(refined_matches1) > MIN_MATCHES:
            src_pts = np.float32([ kp_model1[m.queryIdx].pt for m in refined_matches1 ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in refined_matches1 ]).reshape(-1,1,2)
            homography1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography1 is not None:
                h, w = model1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst1 = cv2.perspectiveTransform(pts, homography1)
                print(pts)
                cnt1 = np.float32([dst1[0][0], dst1[1][0], dst1[2][0], dst1[3][0]])
                frame = cv2.polylines(frame, [np.int32(dst1)], True, 255, 3, cv2.LINE_AA) 
        
        else:
            print "Not enough matches found in 1- %d/%d" % (len(refined_matches1), MIN_MATCHES)
        
        #homography     
        homography2 = None
        if len(refined_matches2) > MIN_MATCHES:
            src_pts = np.float32([ kp_model2[m.queryIdx].pt for m in refined_matches2 ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in refined_matches2 ]).reshape(-1,1,2)
            homography2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography2 is not None:
                h, w = model2.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst2 = cv2.perspectiveTransform(pts, homography2)
                cnt2 = np.float32([dst2[0][0], dst2[1][0], dst2[2][0], dst2[3][0]])
                frame = cv2.polylines(frame, [np.int32(dst2)], True, 255, 3, cv2.LINE_AA) 

        else:
            print "Not enough matches found in 2- %d/%d" % (len(refined_matches2), MIN_MATCHES)
           
        #code for ping pong 
       #print(c)
       # print(tuple(c))
        cv2.circle(frame,(int(c[0]),int(c[1])), 30, (0,0,255), -1) # draw a circle
        c[0] = c[0] + 10*d[0] #shift circle center
        c[1] = c[1] + 10*d[1] #shift circle center
        col1 = False
        col2 = False
        print(check,'ll',d)
        '''if (c[0] > width or c[0] < 0): 
            print('jj')
            break #break if ball moves out
        if (c[1] > height or c[1] < 0): 
            print('tt')
            break #break if ball moves out'''
        #if not hc1 is None : 
        col1 = check_distance(c, d, cnt1) #check if within contour
        #print('ll')
        #if not hc2 is None : 
        col2 = check_distance(c, d, cnt2) #check if within contour
        #print('mm')
        #print(col1,col2)
        if (col1 and check==False): 
            check=True
            d = changedir(d, camera_parameters, homography1,np.mean(dst1,axis = 0),c) #change direction
        elif (col2 and check==False): 
            check=True
            d = changedir( d, camera_parameters, homography2,np.mean(dst2,axis = 0),c)#change direction
        
        if(col1==False and col2==False):
            check=False;
        out.write(frame)
        cv2.imshow('window',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
if __name__ == '__main__':
    main()
