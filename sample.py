import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

arr=np.zeros((650,650),dtype =np.uint8)

cv.imshow("black",arr)
arr1=(np.zeros((650,650),dtype =np.uint8)+255)

cv.imshow("white",arr1)
arr2=(np.zeros((650,650),dtype =np.uint8)+120)

cv.imshow("grey",arr2)

merge=cv.merge((arr,arr1,arr2))
cv.imshow("merged",merge)


cv.waitKey(0 )
cv.destroyAllWindows()
print(merge.shape)
