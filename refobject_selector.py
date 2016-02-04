"""program selects reference objects to follow for tracking keyboard"""

import cv2
  
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
  
def select_reference(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    image = param
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
  
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
  
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Select Objects", image)
         
# construct the argument parser and parse the arguments
def choose(image):
      
# load the image, clone it, and setup the mouse callback function
 
    clone = image.copy()
    cv2.namedWindow("Select Objects")
    cv2.setMouseCallback("Select Objects", select_reference, image)
    selected_refs = list()
    centres = list()
# keep looping until the 'q' key is pressed
    while True:
    # display the image and wait for a keypress
        cv2.imshow("Select Objects", image)
        key = cv2.waitKey(1) & 0xFF
  
    # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            #need to reset the callback as the pointer to image has changed
            cv2.setMouseCallback("Select Objects", select_reference, image)  
    # if the 's' key is pressed, save the image as one of the reference
        elif key == ord("s"):
            if len(refPt) == 2:
                ref = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                selected_refs.append(ref)
                centre = ((refPt[0][0]+refPt[1][0])/2,(refPt[0][1]+refPt[1][1])/2)
                centres.append(centre)
                 
                cv2.imshow("%d" % len(selected_refs), ref)
                key = cv2.waitKey(0) 
 
        elif key == ord("c"): break
# if there are two reference points, then crop the region of interest
# from the image and display it
    '''if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("Ref", roi)
        key = cv2.waitKey(0)'''
         
# close all open windows
    cv2.destroyAllWindows()
    image = clone
    return selected_refs, centres
