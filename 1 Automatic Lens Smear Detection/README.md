
# Homework 1: Automatic Smear Detection
Students: Zunran Guo & Feiyu Chen 

Video demo: smear_detect_result.avi  
PDF Report: Report.pdf  
PPT Report: See this [google slide](https://docs.google.com/presentation/d/1IalqFpkxJtC_u9LaJZ0p9Aec7sGGh-wenFx8FT9bePM/edit?usp=sharing).


# Code: Method 1
Open and run "main_method1.ipynb".  
It reads in the [image.png](image.png) and detect smear on it by using thresholding and contour analysis algorithms.

# Code: Method 2
Run "main_method2.py" by:   
$ python3 main_method2.py

It reads in the images from "cam_3/" folder and display the smear detection result starting from the 500th frame. The algorithm uses a window size of 500 frames to compute the variation of pixel values, then use Method 1 to detect smears.  
(Before running code, you need to put the homework dataset "cam_3/" in this folder.)
