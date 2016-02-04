"""program selects markers for keyboard"""

import cv2
  
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
  
def select_reference(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    image = param
    # if the left mouse button was clicked, record the
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image,(x,y),1,(0,255,0),2)
        cv2.imshow("Select Markers", image)
 
         
# construct the argument parser and parse the arguments
def choose(image):
      
# load the image, clone it, and setup the mouse callback function
 
    clone = image.copy()
    cv2.namedWindow("Select Markers")
    cv2.setMouseCallback("Select Markers", select_reference, clone)
    selected_refs = list()
# keep looping until the 'q' key is pressed
    while True:
    # display the image and wait for a keypress
        cv2.imshow("Select Markers", image)
        key = cv2.waitKey(1) & 0xFF
  
    # if the 'r' key is pressed, reset the points
        if key == ord("r"):
            image = clone.copy()
            #need to reset the callback as the pointer to image has changed
            cv2.setMouseCallback("Select Markers", select_reference, image)
    # if the 's' key is pressed, save the image as one of the reference
        elif key == ord("s"):
            selected_refs = refPt
            break
 
 
         
# close all open windows
    cv2.destroyAllWindows()
    return selected_refs
