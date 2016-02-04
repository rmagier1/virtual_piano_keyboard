"""File with all the helped function of our project"""
import cv2
import numpy as np
import pickle
from copy import deepcopy
import cv2.cv


import object_selector as o
import refobject_selector as r
import tracker as tk



##############################################################################
"""function chooses the markers and reference images if choose is True and 
saves them for later use if needed, 
otherwise it will get the files previously saved"""
def setVars(choose, ref_image):

    if choose:
        markers = o.choose(ref_image)
        marks_file= open('dmarkers.pk', 'w')
        ref_file = open('dref.pk','w')
        ref_c_file = open('dcen.pk','w')
        cv2.imwrite('dref_image.jpg', ref_image)
        reflist, ref_centers = r.choose(ref_image)

        pickle.dump(markers, marks_file)	#save segmentation for later use
        pickle.dump(reflist, ref_file)
        pickle.dump(ref_centers, ref_c_file)

        marks_file.close()			#close files
        ref_file.close()
        ref_c_file.close()
    else:
        ref_image=cv2.imread('dref_image.jpg')	#load stored variables
        marks_file= open('dmarkers.pk', 'r')
        ref_file = open('dref.pk','r')
        ref_c_file = open('dcen.pk','r')
        markers = pickle.load(marks_file)
        reflist = pickle.load(ref_file)
        ref_centers = pickle.load(ref_c_file)

    return markers, reflist, ref_centers
##############################################################################

""" get_keys_from_markers
    given an image of a keyboard and a list of x,y coordinates as markers
     the function uses the watershed algorithm to return a mask with pixels labelled
    positive integers for regions matching the markers and -1 elsewhere
"""
def get_keys_from_markers(img, markers_list):
 
    #make the markers in the format required by the algorithm
    h, w = img.shape[:2]
    markers_array = np.zeros((h, w), np.int32)
    boundary = 0
    for i, mark in enumerate(markers_list):
 
        markers_array[mark[1]][mark[0]] = i+1
        boundary +=1
    cv2.watershed(img, markers_array)
    return markers_array, boundary
###############################################################################

"""rescale images to be all equal size"""
def scale_image(ref_image,  marks,ref_shad):
    ref_image = cv2.resize(ref_image, ref_shad.shape[:2])

    #numpy coords are around the other way for marks
    marks = cv2.resize(marks, (ref_shad.shape[1], ref_shad.shape[0]), interpolation=cv2.INTER_NEAREST)
    return ref_image, marks
###############################################################################

def get_centers(reflist, ref_markers, frame, size=0.5):
    refdeskps = list()
    for ref in reflist:
        refdeskps.append(tk.scan_ref(ref))
    scenekps, scenedes = tk.scan_scene(frame, size)
    matches_kp = list()
    for r in refdeskps:
        refdes = r[1]
        refkps = r[0]
        matches_kp.append(tk.f_match(refdes, refkps, scenedes, scenekps))
 
    #get transforms
    transforms = list()
    ref_kp_list_xy = list()
    scene_kp_list_xy = list()
 
    for m in matches_kp:
 
        if len(m) < 2:
            transforms.append(None)
            continue
 
 
        ref_xy, scene_xy = tk.split_kps_to_ref_frame(m)
 
        transforms.append(tk.transform(ref_xy, scene_xy))
        ref_kp_list_xy.append(tk.kp_to_xy(ref_xy, 1))
        scene_kp_list_xy.append(tk.kp_to_xy(scene_xy, size))
 
 
    #get corners of objects in the scene
    centers = list()
    for idx, ref in enumerate(reflist):
        center = tk.get_center(ref, transforms[idx],size)
        centers.append(center)
    return centers
############################################################################
def transform_marks(o_centers, new_centres, markers, shape, size=0.5):
    bounds = getArea(new_centres, shape)
    if new_centres is None: return markers
    o_centers = centre_to_array(scale_centers(o_centers, size))
    new_centres = centre_to_array(scale_centers(new_centres, size))
    trs = cv2.estimateRigidTransform(o_centers, new_centres,True)

    if trs is None: return markers
    new_markers = np.zeros(markers.shape, dtype='uint8')
    #downsize for computation
    small_new = cv2.resize(new_markers, (0,0),fx=size, fy=size )
    small_markers = cv2.resize(markers, (0,0),fx=size, fy=size)
 
 
    for x in range(int(bounds[2]*size),int(bounds[3]*size)):
        for y in range(int(bounds[0]*size),int(bounds[1]*size)):
            try:
                if small_markers[x][y] <= 0: continue
            except: pass
            c = np.array([[x],[y],[1]])
 
            v = np.dot(trs,c)
            if v[0]>=0 and v[0] < len(small_markers) and v[1]>=0 and v[1] < len(small_markers[0]):
                small_new[int(v[0]),int(v[1])] = small_markers[x][y]
    new_markers = cv2.resize(small_new, (0,0),fx=1/size, fy=1/size, interpolation = cv2.INTER_NEAREST)
    return new_markers
###########################################################################
 
def getArea(centres, shape):
    left, right, up, down = float('inf'), 0, float('inf'), 0
    try:
        for c in centres:
            if c[0] < left: left = c[0]
            if c[0] > right: right = c[0]
            if c[1] > down: down = c[1]
            if c[1] < up: up = c[1]
    except: left, right, up, down = 0, shape[1]-1, 0, shape[0]-1
    return (left, int(right), up, int(down))
##########################################################################
 
def centre_to_array(centres):
 
    new_centres = []
    for x in centres:
        new_centres.append([x[0],x[1]])
    return np.array([new_centres],dtype='float32')
##########################################################################
def scale_centers(centers, size):
    new = list()
    for c in centers:
        new.append((int(c[0]*size), int(c[1]*size)))
    return new
############################################################################
