'''
Created on 4 Apr 2015

@author: Rebecca and Sean
'''
import cv2
import cv2.cv
import numpy as np

def scan_ref(ref):
    #scans the reference image and appends kps and descriptors to self.kps
    ref = ref.copy()
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(ref, None)
    return kp, des
    
        
def scan_scene(scene, size):
    #size is the sampling to take. eg 0.5 to halve image size
    img2 = cv2.resize(scene,(0,0), fx=size, fy=size)
    #img2 = cv2.GaussianBlur(img2,(5,5),0)
    
    sift = cv2.SIFT()
    scenekp, scenedes = sift.detectAndCompute(img2, None)
    return scenekp, scenedes
def f_match(refdes, refkps,scenedes, scenekps):
    #matches the features of a set of reference images to a scene
    #returns the matched key points indexes
    #pass the kp and desc of ref so we don't recompute every frame
    
    matches = list()    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(refdes, scenedes, k=2)
    
        
        # ratio test as per Lowe's paper
    kp = list()
       
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            kp.append((refkps[m.queryIdx], scenekps[m.trainIdx])) 
    #append the keypoint  matches to list of kps
    
    return kp
   
def get_all_frame_kp(video):
    kps = list()
    while(True):
        ret, frame = video.read()
        if not ret: break
        kps.append(f_match(frame))
    
def kp_to_xy(kps, size):
    new = list()
    for k in kps:
        new.append((int(k.pt[0]/size), int(k.pt[1]/size)))
    return new

def kp_to_xy_for_transform(kps):
    new = list()
    for k in kps:
        new.append([int(k.pt[0]), int(k.pt[1])])
    return new
    
def split_kps_to_ref_frame(kps):
    if kps is None: raise Exception("No key points, need to run the matcher first")
    refkp = list()
    framekp = list()
    for x in kps:
        refkp.append(x[0])
        framekp.append(x[1])
    return refkp, framekp
    

def write_video(video):
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
    out  = cv2.VideoWriter('thirdsize.avi',-1,fps,(width, height),3)
        
    while(True):
        ret, frame = video.read()
        if not ret: break
        f_match(frame)
            
        #########insert Display function here
        out.write(frame)
    
def get_first_frame(video):
    ret, frame = video.read()
    return frame

def transform(refkp, framekp):
    #add the extra [] to make 3D array
    
    if refkp is None: return None
    if framekp is None: return None
    #if less than 5 matches then we can't find a transform
    if len(refkp) < 5: return None
    
    refkp = np.array([kp_to_xy_for_transform(refkp)],dtype = 'float32')
    framekp = np.array([kp_to_xy_for_transform(framekp)], dtype = 'float32')
    trs = cv2.estimateRigidTransform(refkp,framekp,False)
    
    return trs

def get_corners(reference, trs,size):
    ul = np.array([[0],[0],[1]])
    ur = np.array([[reference.shape[1]],[0],[1]])
    br = np.array([[reference.shape[1]],[reference.shape[0]],[1]])
    bl = np.array([[0],[reference.shape[0]],[1]])
    corners = [ul, ur, br, bl]
    #make integer for display
    if (trs is None): return list()
    if not trs.any(): return list()
    new_corners = list()
    for point in corners:
        
        point = np.dot(trs,point)
        new_corners.append((int(point[0]/size),int(point[1]/size)))

    
    return new_corners
#returns the center given the trs
def get_center(reference, trs,size):
    c = np.array([[reference.shape[1]/2],[reference.shape[0]/2],[1]])
    if (trs is None): return None
    if not trs.any(): return None
    centre = np.dot(trs,c)
    return (int(centre[0]/size),int(centre[1]/size))