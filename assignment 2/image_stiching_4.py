import cv2 as cv
import numpy as np
import glob as glob
import copy
from matplotlib import pyplot as plt
import sys

def readImages(folder):
    files = glob.glob(folder + "/*.jpg")
    for img in files:
        print img
    images = [cv.imread(img) for img in files]
    return images

def resizeImages(images):
    resized = []
    for img in images:
        h,w = img.shape[:2]
        r_img = cv.resize(img,(int(w/4),int(h/4)))
        r_img = cv.cvtColor(r_img,cv.COLOR_BGR2GRAY)
        resized.append(r_img)
    return resized
    
def getKeyPoints(images):
    surf_obj = cv.xfeatures2d.SURF_create()
    kp_des = [surf_obj.detectAndCompute(img,None) for img in images]
    return kp_des
        
def matchFeatures(kp_des):
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)
    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = [[0 for i in range(len(kp_des))] for j in range(len(kp_des))]
    print matches
    for i in range(len(kp_des)):
        for j in range(len(kp_des)):
            if i == j:
                matches[i][j] = len(kp_des[i][0])
            else:
                temp_match = flann_matcher.knnMatch(kp_des[i][1],kp_des[j][1],k=2)
                temp_match = refineMatch(temp_match)
                matches[i][j] = temp_match
    return matches
    
def refineMatch(match):
    f_match = []
    for m,n in match:
        if m.distance < 0.7*n.distance:
            f_match.append(m)
    return f_match

def getUCHomography(matches,kp_des):
    H = [[-1 for i in range(len(kp_des))] for j in range(len(kp_des))]
    for i in range(len(matches)):
        for j in range(len(matches)):
            if i == j:
                H[i][j] = iMatrix()
            else:
                matchij = matches[i][j]
                h = getSingleHomography(i, j, matches, kp_des)
                H[i][j] = h
                
    return H
    
def getSingleHomography(i, j, matches, kp_des):
    if len(matches[i][j])>10:
        src_pts = np.float32([ kp_des[i][0][m.queryIdx].pt for m in matches[i][j] ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_des[j][0][m.trainIdx].pt for m in matches[i][j] ]).reshape(-1,1,2)
        h, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,3.0)
        return h
    else:
        return -1            

def getCompleteHomography(H):
    change = True
    while(change):
        change = False
        for i in range(len(H)):
            for j in range(len(H)):
                if not type(H[i][j])==np.ndarray:
                    change = True
                    h = getPsuedoHomography(i,j,H)
                    H[i][j] = h
    return H

def getPsuedoHomography(i,j,H):
    for k in range(len(H)):
        if type(H[i][k])==np.ndarray and type(H[k][j])==np.ndarray:
            h = H[k][j].dot(H[i][k])
            return h
    return -1
    
def getSequence(c, matches):
    m, seq = [], []
    for i in range(c):
        m.append(len(matches[i][c]))
    m.append(-1)
    for j in range(c+1,len(matches)):
        m.append(len(matches[c][j]))
    m_sort = copy.deepcopy(m)
    m_sort.sort()
    for i in range(len(m)):
        seq.append(m.index(m_sort[i]))
    return seq
    
def findBestMatch(matches):
    m =  len(matches[0][1])
    pos = [0, 1]
    for i in range(len(matches)):
        for j in range(len(matches)):
            if not i==j:
                print i, j
                print len(matches[i][j])
                if len(matches[i][j]) > m:
                    pos = [i, j]
                    m = len(matches[i][j])
    return pos
   
def findNextBest(rem, done, matches):
    m = -1
    pos = [-1, -1]
    ordr = 0
    for i in range(len(done)):
        for j in range(len(rem)):
            if done[i]<rem[j]:
                if len(matches[done[i]][rem[j]]) > m:
                    pos = [done[i], rem[j]]
                    m = len(matches[done[i]][rem[j]])
                    ordr = 1
            elif done[i]>rem[j]:
                if len(matches[rem[j]][done[i]]) > m:
                    pos = [rem[j], done[i]]
                    m = len(matches[rem[j]][done[i]])
                    ordr = -1
    if ordr == -1:
        done.append(pos[0])
        rem.remove(pos[0])
        return pos[0]
    if ordr == 1:
        done.append(pos[1])
        rem.remove(pos[1])
        return pos[1]

def mergeImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    warped_image = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin),flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    warped_image[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return warped_image
    
def mergeImages2(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    print h1, w1
    print h2, w2
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)
    print pts1
    print pts2_
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    print xmax, ymax, xmin, ymin
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    warped_image = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin),flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    print "t1 = " + str(t[1])
    print "t0 = " + str(t[0])
    warped_image[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return warped_image
    
def warpImages(images, seq, app_h):
    h, w, pts, pts_, ptst = [], [], [], [], []
    for i in range(len(seq)):
        h.append((images[i].shape)[0])
        w.append((images[i].shape)[1])
        p = np.float32([[0,0],[0,h[i]],[w[i],h[i]],[w[i],0]]).reshape(-1,1,2)
        pts.append(p)
        pts_.append(cv.perspectiveTransform(pts[i], app_h[i]))
    ptst = np.concatenate((pts_[0], pts_[1]), axis=0)
    print h
    print w
    print pts
    print pts_
    for j in range(2,len(seq)):
        ptst = np.concatenate((ptst, pts_[j]), axis=0)
    print ptst
    [xmin, ymin] = np.int32(ptst.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(ptst.max(axis=0).ravel() + 0.5)
    print xmax, ymax, xmin, ymin
    t = [-xmin,-ymin] 
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    warped_images = [0 for k in range(len(images))]
    panorama = np.zeros(shape=[ymax-ymin, xmax-xmin], dtype=np.uint8)
    for i in range(len(images)):
        warped_images[i] = cv.warpPerspective(images[i], Ht.dot(app_h[i]), (xmax-xmin, ymax-ymin),flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return warped_images

def mergeImages(images):
    h, w, d = images[0].shape
    panorama = np.zeros(shape=[h, w, d], dtype=np.uint8)
    num = 0
    val = 0
    for i in range(h):
        for j in range(w):
            for k in range(d):
                print i, j
                for img in images:
                    if not img[i][j][k]==0:
                        val = val + img[i][j][k]
                        num = num+1
                if num!=0 and val!=0:       
                    panorama[i][j][k] = int(val/num)
                num = 0
                val = 0
               
    return panorama
    
def iMatrix():
    i = np.float32([[1,0,0],[0,1,0],[0,0,1]])
    print i
    return i
    
def getColorResizedimages(images):
    cimg = []
    for img in images:
        h,w = img.shape[:2]
        r_img = cv.resize(img,(int(w/4),int(h/4)))
        r_img = cv.cvtColor(r_img,cv.COLOR_BGR2HSV)
        cimg.append(r_img)
    return cimg
        

def createPanorama(folder):
    color_images = readImages(folder)
    images = resizeImages(color_images)
    kp_des = getKeyPoints(images)
    matches = matchFeatures(kp_des)
    H = getUCHomography(matches,kp_des)
    H_comp = getCompleteHomography(H)
    rem = [i for i in range(len(images))]
    seq, app_h = [], []
    a, b = findBestMatch(matches)
    base_image = b
    seq.append(b)
    app_h.append(iMatrix())
    seq.append(a)
    app_h.append(H_comp[a][b])
    rem.remove(a)
    rem.remove(b)
    while(len(rem) > 0):
        c = findNextBest(rem, seq, matches)
        app_h.append(H_comp[c][b])
    print seq
    print app_h
    
    '''for i in range(len(seq)):
        warped_img_2 = mergeImages2(images[b], images[seq[i]], app_h[i])
        plt.imshow(warped_img_2, 'gray'),plt.show()'''
    cimg = getColorResizedimages(color_images)
    warpedImages = warpImages(cimg, seq, app_h)
    panorama = mergeImages(warpedImages)
    plt.imshow(panorama, 'gray'),plt.show()
panorama = createPanorama(sys.argv[1])
cv.imwrite("output.jpg", panorama)