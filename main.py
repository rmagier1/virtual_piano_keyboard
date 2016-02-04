"""Authors: Sean Benson and Rebecca Magier"""

#imported files
import cv2
import numpy as np
import pickle
from copy import deepcopy
import functions as func
import hand_detector as hd
import Tkinter
import tkSnack
import playnotes

frame = 0
frequencies = [175,196,220,247,262, 294, 330,350, 392,440,494,523, 587, 659, 698, 784, 880, 988, 1047]

root = Tkinter.Tk()
tkSnack.initializeSnack(root)
choose = True
tracking = False
vfile = 0
draw_contours = False


colors = np.int32(list(np.ndindex(3, 3, 3))) * 127 #colors to segment keyboard
cap = cv2.VideoCapture(vfile) #read in video or open camera
ret, ref_shad = cap.read()	#first frame is reference image
markers, reflist, ref_centers = func.setVars(choose,ref_shad)  #set variables
ref_image=cv2.imread('dref_image.jpg')		#have untouched ref image

#Segment the keyboard
marks, boundary = func.get_keys_from_markers(ref_image,markers)
overlay = colors[np.maximum(marks, 0) % 27]
vis = cv2.addWeighted(ref_image, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
cv2.imshow('vis', vis)	#display segmentation

#rescale image to all be equal size
ref_image, marks = func.scale_image(ref_image,marks.view('uint8')[:,::4],ref_shad)

cv2.waitKey(0)			#waits for user to exit vis screen

#set up writing video to output file
fourcc = cv2.cv.CV_FOURCC(*'DIVX')
fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
height, width, depth = ref_image.shape
#out = cv2.VideoWriter('real_track.avi',-1,24,(height, width))


#start main video loop
ret = True
prev_tips = None
prev_pressed = None

playing = list()
while(ret):
    frame +=1
    ret, img = cap.read()	#read in next frame
    if(not ret): break
    h, w = img.shape[:2]

    skin, mask = hd.skin_detector(img)
    shadow = hd.get_shadow(img, ref_shad, mask)
    tips, contours = hd.get_tips(skin)

    if tracking:
    #record all the values
        centers = func.get_centers(reflist, markers, img)

    #only use detected centers
        detected_refs = list()
        for i, c in enumerate(centers):
            if c is not None:
                detected_refs.append(ref_centers[i])
        for c in centers:
            if c is None:
                centers.remove(c)
 
        for c in centers:
            cv2.circle(img, c, 4, (255,255,0), 2)
        num_centers = len(centers)
        for x in range(num_centers):
            centers.append(centers[x])
            detected_refs.append(detected_refs[x])
 
 
    #transform the markers
        if frame %1 == 0: new_board = func.transform_marks(detected_refs, centers, marks, ref_shad.shape[:2])
        else: new_board = marks
    else: new_board = marks
    overlay = colors[np.maximum(new_board, 0) % 27]
    #vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
    pressed = hd.keys_pressed(tips, prev_tips, marks, boundary, shadow)
    prev_tips = tips
    play = list()
    pressed_marks = np.empty((h, w), dtype=np.int32)
    pressed_marks.fill(-1)
    print pressed
    for p in pressed:

        #in not pressed, remove from playing
        for key in playing:
            if key not in pressed or not prev_pressed: playing.remove(key)
        if p in prev_pressed:
            #add visuals
            seg = np.where(new_board==p)
            pressed_marks[seg]=p

            if p not in playing:
                play.append(p)
                playing.append(p)
    prev_pressed = pressed

    if play:
        for key in play:
            playnotes.playNote(frequencies[key], 0.25)

    if draw_contours:		#draw the contours if specified

        try:
            for c in range(len(contours)):
                cv2.drawContours(img, contours, -1, (0,255,0),3)
            for tip in tips:
                cv2.circle(img, tip, 4, (255,100, 255), 2)
        except: pass
    #overlay = colors[np.maximum(new_board, 0) % 27]
    overlay = colors[np.maximum(pressed_marks, 0) % 27] * 255
    vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
    #cv2.imshow('shad', shadow)
    cv2.imshow('img', vis)
    #out.write(vis)
    '''out.write(img)'''
    wkey = cv2.waitKey(1) & 0xFF
    if wkey == ord('q'):
        #out.release()
        ret = False
        break
#out.release()
playnotes.soundStop()

