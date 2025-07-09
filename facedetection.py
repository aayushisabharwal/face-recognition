import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('photos/group.jpg')
cv.imshow('group', img)
# haar cascades are very sensitive to noise 
#convert to gray scale : harr looks at the object and not the color present 
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray person', gray)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# we can reduce the senstivity by changing  the scale factor and min neighbours
# by changing minneighbours -- reducing it it becomes more prone to noise
faces_rect = haar_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=6)

print(f'number of faces found ={len(faces_rect)}' )

# faces (sometimes named faces_rect or faces_rects) is a list of rectangles.
# Each rectangle describes where the face is in the image.#
# x = X-coordinate of the top-left corner of the rectangle (horizontal position).
# y = Y-coordinate of the top-left corner (vertical position).
# w = width of the rectangle (how wide the face area is).
# h = height of the rectangle (how tall the face area is).

#+----------------------------+
#|                            |
#|        (x,y) *-------------+    
#|               |            |    
#|               |  FACE      |    
#|               |            |    
#|               +-------------*    
#|                            |
#+----------------------------+

for(x,y,w,h) in faces_rect:
    cv.rectangle(img , (x,y) , (x+w , y+h) , (0,255,0) , thickness=2)

cv.imshow('dected faces', img)
cv.waitKey(0)