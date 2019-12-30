# coding: utf-8
import cv2
import numpy as np
from numpy.linalg import inv

# Load n color image
img = cv2.imread('input.jpg')

p1 = np.array([[31,186],[166,186],[234,86],[136,81]])
p2 = np.array([[45,525],[188,525],[188,174],[45,174]])

H,t = cv2.findHomography(p1,p2)
print("Homography Matrix:",H)

player1_pos = np.array([[162],[233],[1]])
player2_pos = np.array([[210],[147],[1]])

pp1 = np.matmul(H,player1_pos)
pp2 = np.matmul(H,player2_pos)

#print(pp1,pp2)

pp1a = pp1 / pp1[-1]
pp2a = pp2 / pp2[-1]

pp2a = np.matmul(inv(H),pp1a)
pp2b = np.matmul(inv(H),pp2a)
pp2a = (pp2a / pp2a[-1])
pp2b = (pp2b / pp2b[-1])

pp2a = pp2a.astype(int)
pp2b = pp2b.astype(int)

ht = img.shape[0]

img = cv2.line(img, (pp2a[1],0), (pp2a[0],ht), (255, 0, 0), 2)
img = cv2.line(img, (pp2b[0], 0), (pp2b[1], ht), (0, 255, 0), 2)

pixels_per_metric = 353 / 18.3
#print(pixels_per_metric)

print("The distance between two lines in meters:", (pp2a[1]-pp2b[0])/pixels_per_metric)

cv2.imshow('Input',img)
cv2.waitKey(0)
cv2.destroyWindow('image')
cv2.waitKey(1)
