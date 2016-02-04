"""program for all hand detector functions"""

import cv2
import numpy as np
from copy import deepcopy


############################################################################
"""function detects the skin and saves the mask"""
def skin_detector(img):
    #color bounds
    lower = np.array([0,20,40], dtype="uint8")
    upper = np.array([30,255,255], dtype="uint8")
    lower2 = np.array([175,20,40], dtype="uint8")
    upper2 = np.array([180,255,255], dtype="uint8")

    img_clone = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  #clone image

    mask = cv2.inRange(img_clone,lower,upper)     #create masks
    mask2 = cv2.inRange(img_clone, lower2, upper2)
    mask = cv2.bitwise_or(mask, mask2)
 
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(mask,kern, iterations = 2)
 
    mask = cv2.dilate(mask,kern, iterations = 2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    skin = cv2.bitwise_and(img, img, mask = mask)
 
    return skin, mask
############################################################################
"""find shadow of hand and return it"""
def get_shadow(image, reference, skinMask):

    image = cv2.GaussianBlur(image, (5,5),0)	#blur images
    reference = cv2.GaussianBlur(reference, (5,5),0)

    invert = cv2.bitwise_not(skinMask)
    maskImage = cv2.bitwise_and(image,image , mask=invert)
    maskRef = cv2.bitwise_and(reference,reference, mask= invert)

    maskImage = maskImage.astype(np.int16)
    maskRef = maskRef.astype(np.int16)

    newImage = maskRef - maskImage
    newImage = newImage.clip(0)
    newImage = newImage.astype(np.uint8)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)

    ret, thres = cv2.threshold(newImage,50,255, cv2.THRESH_BINARY)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    return thres
###########################################################################
"""use the image of the skin to locate fingertips
 assumes that the hand is pointed down towards the bottom of the image."""
def get_tips(skin):
    thres = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    tips = []
    ret, thres = cv2.threshold(thres,1,255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    try:
        for c in range(len(contours)):
            minps = local_min(contours[c])
            for minp in minps:
                point = (minp[0], minp[1])
                tips.append(point)
                    
    except: pass
    return tips, contours
###########################################################################
"""find local min using derivative"""
def local_min(contour):
    min = 0
    prev_dy = None
    min_flag = False
    temp_min_points = list()
    min_points = list()
    for i in range(len(contour)-1):
        dy = contour[i+1][0][1] - contour[i][0][1]
        #case starting a min or continuing
        if (prev_dy > 0 and dy <= 0) or (min_flag and dy == 0):
            min_flag = True
            miny = contour[i][0][1]
            temp_min_points.append([contour[i][0][0], contour[i][0][1]])
 
        #case we didn't have a min only a step
        elif min_flag and dy > 0:
            min_flag = False
            temp_min_points = []
        #case gone up bu not by much
        elif min_flag and contour[i][0][1] == miny:
            temp_min_points.append([contour[i][0][0], contour[i][0][1]])
 
        #case we are exiting min
        elif dy < 0 and min_flag:
 
            points = np.array(temp_min_points)
            xvals = points[:,0]
            min_points.append([int(np.mean(xvals)),miny])
            min_flag = False
            temp_min_points = []
        prev_dy = dy
    return min_points
##############################################################################
def keys_pressed(tips, prev_tips, markers, boundary, shadow):
    THRESHOLD = 4
    keys_pressed = []
    #loop through tips, see if on a marker. If so, go up key to find length to the end of the shadow. Key is pressed if distance below threshold
    for tip in tips:
        if locate_tip(tip, markers, boundary) > 0:
            if tip_displacement(tip, prev_tips)[1] <= THRESHOLD and tip_displacement(tip, prev_tips)[1] >= -THRESHOLD and shadow_length(tip, shadow) < 10 and markers[tip[1], tip[0]] not in keys_pressed:
                keys_pressed.append(markers[tip[1], tip[0]])
    return keys_pressed

def locate_tip(tip, marks, boundary):
    if marks[tip[1], tip[0]] > 0 and marks[tip[1], tip[0]] < boundary-1:
        return marks[tip[1], tip[0]]
    else: return 0


def shadow_length(tip, shadow):
    #returns the length of the shadow under the tip
    length = 0
    if tip == None: return None
    while shadow[tip[1]+length+1, tip[0]] > 0:
        length += 1
    return length
#returns a vector of the displacement of the previous tip location
#looks in an area 14x100 pixels and chooses closest


def tip_displacement(tip, previous_tips):
    X, Y = 7, 50
    min = X**2 + Y**2
    closest = (1000,1000)
    try:
        for p in previous_tips:
            if abs(tip[0]-p[0]) < X and abs(tip[1] - p[1]) < Y:
                if ((tip[0]-p[0])**2 + (tip[1] - p[1])**2) < min:
                    closest = (tip[0]-p[0], tip[1] - p[1])
    except: pass
    return closest

#returns the amount of shadow pixels on a key
def shadow_percentage(key, markers, shadow):
    marks_copy = deepcopy(markers)
    mask = (marks_copy != key)
    #set all pixels outside region to zero
    marks_copy[mask] = 0

    segment_px_idxs = np.nonzero(marks_copy)
    key_pixels_count = np.count_nonzero(marks_copy)

    shadow_pixels_count = np.count_nonzero(shadow[segment_px_idxs])
    return float(shadow_pixels_count)/key_pixels_count

def get_ml_values(key, tips, prev_tips, shadow, marks, boundary):
    tip = tip_for_key(tips, key, marks, boundary)
    s = "tip dis = "+str(tip_displacement(tip, prev_tips))

    s = s+" shad perc = "+ str(shadow_percentage(key, marks, shadow))
    s= s+" shad left "+str(shadow_percentage(key-1, marks, shadow))
    s= s+" shad right "+str(shadow_percentage(key+1, marks, shadow))
    s= s+" shad above "+str(shadow_above_tip(tip, shadow))
    s= s+" shad below "+str(shadow_below_tip(tip, shadow))
    s= s+" shad length"+str(shadow_length(tip, shadow))
    return s

def tip_for_key(tips, key, marks, boundary):
    for tip in tips:
        if locate_tip(tip, marks, boundary) == key: return tip
    return None

def shadow_below_tip(tip, shadow):
    count = 0
    if tip == None: return None
    for x in range(20):
        for y in range(20):
            if shadow[tip[1]+y][tip[0]-10+x] > 0: count+=1
    return count

def shadow_above_tip(tip, shadow):
    count = 0
    if tip == None: return None
    for x in range(20):
        for y in range(20):
            if shadow[tip[1]-y][tip[0]-10+x] > 0: count+=1
    return count