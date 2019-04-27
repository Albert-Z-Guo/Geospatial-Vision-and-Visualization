
'''
# What does this file do?
    Find static regions in the video.
    Detect smears among these regions.
    Display the result.

# How to run this file:
    Put the homework's dataset "cam_3/" at the same level as this file. 
    Then, run this file by python3.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections


def create_folder(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def gray2color(I):
    return cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)


def show_result_image(cnt_img, images_list):
    # Combine 4 images in images_list, and display the result.
    # images_list: [src_img, gray2color(background_img), resI_objects, resI_smear_on_src]

    # Put string onto each image
    def int2str(num, blank): return ("{:0"+str(blank)+"d}").format(num)
    titles = ["", "white pixel changes less",
              "pixel value thresholding", "detected smears"]
    for i in range(4):
        prefix = "Image " + int2str(cnt_img, 4) if i == 0 else ""
        cv2.putText(
            images_list[i], text=prefix+titles[i], org=(20, 80),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1.0, color=(0, 0, 255), thickness=2)

    # Stack images into one image
    i1 = np.hstack(tuple(images_list[0:2]))
    i2 = np.hstack(tuple(images_list[2:4]))
    img_show = np.vstack((i1, i2))
    img_show = cv2.resize(img_show, dsize=(0, 0), dst=None, fx=0.7, fy=0.7)
    cv2.imshow('Smear detection result', img_show)

    # Wait for keypress
    k = cv2.waitKey(10)
    key_break = False
    if k != -1 and chr(k) == 'q':
        key_break = True
    return key_break, img_show


class ImageReader(object):  # A class for reading image
    def __init__(self, source, folder_name=None, video_name=None):
        assert source in ["from_video", "from_folder"] # Choose source from the two

        if source == "from_video":
            self.video_name = video_name
            self.cap = cv2.VideoCapture(self.video_name)
        else:  # from_folder
            self.folder_name = folder_name + \
                "/" if folder_name[-1] != "/" else folder_name
            self.filenames = self.get_filenames(self.folder_name)

        self.source = source
        self.cnt_image = 0

    def __del__(self):
        if self.source == "from_video":
            self.cap.release()

    def read(self, filename=None, cap=None, resize_rate=1):

        if self.source == "from_video":
            ret, frame = self.cap.read()
            assert not ret, "All images in video were read out"
        else:  # from_folder
            frame = cv2.imread(self.filenames[self.cnt_image])

        self.cnt_image += 1

        if resize_rate != 1: # Resize image
            frame = cv2.resize(src=frame, dsize=(
                0, 0), dst=None, fx=resize_rate, fy=resize_rate)

        return frame

    def is_next_image_available(self):
        if self.source == "from_video":
            return True
        else:  # from_folder
            return self.cnt_image < len(self.filenames)

    def get_prev_img_filename(self):
        return self.filenames[self.cnt_image-1].split('/')[-1]

    def get_filenames(self, path, format="*"):
        import glob
        list_filenames = sorted(glob.glob(path+'/'+format))
        return list_filenames  # Path to file wrt current folder


class MotionEstimator(object):  # Detect the variation of each pixel in a video
    def __init__(self):
        self.cnt_processing = 0

    def calc_variation_image(self, imgs_list):
        # rtype: A gray image of np.array of uint8.
        # Moving regions are dark. Static regions are white, which are smears

        # Sum up the absolute differences between each pair of adjacent images
        # to obtain total variation.
        # Scale the variation value to [0, 255], and ouput a gray image.
        # self.variation_imgs: Deque of images. Store number=[len(imgs_list)-1] previous variation images
        # self.variation_img: sum(self.variation_imgs)

        if self.cnt_processing == 0:  # add differences of all past images
            self.variation_imgs = collections.deque()
            self.variation_img = np.zeros(
                imgs_list[0].shape[0:2], dtype=np.float)
            prev = imgs_list[0]
            for i in range(1, len(imgs_list)):
                curr = imgs_list[i]
                diff = np.abs(curr - prev)
                self.variation_img += diff
                self.variation_imgs.append(diff.copy())
                prev = curr
        else:  # add the new difference
            diff = np.abs(imgs_list[-1] - imgs_list[-2])
            self.variation_img += diff
            self.variation_img -= self.variation_imgs.popleft()
            self.variation_imgs.append(diff)
        self.cnt_processing += 1

        print("variation image: max = {:.1f}, min = {:.1f}".format(
            self.variation_img.max(), self.variation_img.min()))

        # Properly scale the variation image to [0, 255]
        res_variation_img = self.variation_img * 5.0 / (len(imgs_list)-1)
        res_variation_img = 255 - res_variation_img.astype(np.uint8)
        #res_variation_img = cv2.equalizeHist(res_variation_img)
        return res_variation_img

    ''' Not used
    def calc_pixel_wise_std(self, imgs_list):
        imgs_list = np.array(imgs_list) 
        std_img = np.std(imgs_list, axis=0)
        # print(std_img.max(), std_img.min())
        std_img *= 3 # properly scale the value to [0, 255]
        std_img = 255 - std_img.astype(np.uint8)
        # print(std_img.shape)
        return std_img
    '''


class SmearDetector(object):  # Detect smear from the variation image
    def __init__(self):
        pass

    def detect(self, src_img, background_img):
        '''
        input:
            background_img: A gray image of np.array of uint8.
                Moving regions are dark. Static regions are white, which are smears.
        Output:
            resI_smear_mask: mask image, indicating the smear.
            resI_objects: a colorful labelled image, indicating each connected component after thresholding
            resI_smear_on_src: the detected smear drawn on source image
        '''

        MIN_INTENSITY = 180  # ! Modify THRESHOLD HERE !
        I_objects = cv2.threshold(
            background_img, thresh=MIN_INTENSITY, maxval=255, type=cv2.THRESH_BINARY)[1]
        I_objects = self.do_morphological_open(I_objects)
        _, labels_img = cv2.connectedComponents(I_objects)

        # Prepare images to plot later
        resI_smear_on_src = src_img.copy()
        resI_objects = self.labels_to_color(labels_img)
        resI_smear_mask = np.zeros((500, 500, 3), np.uint8)

        # Check if each region is a smear
        for i in range(labels_img.max()+1):

            # Get ith object's image
            obj_i = (labels_img == i).astype(np.uint8)*255
            if 0:  # skip too small region
                if np.count_nonzero(obj_i > 0) < 100:
                    continue

            # Detect contour
            contours, _ = cv2.findContours(  # See: https://blog.csdn.net/sunny2038/article/details/12889059
                obj_i, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contour = self.find_longest_contour(contours)
            if len(contour) < 10:  # skip too small region
                continue

            # Check if it's a smear by analysing the contour
            is_smear, area, circularity = self.is_region_large_and_circular(
                contour)

            # Draw
            cv2.drawContours(  # draw each object and contours
                resI_objects, contours=[contour], contourIdx=-1, color=(255, 255, 255), thickness=2)

            if is_smear:  # draw smear
                # print("{}th object in image is a smear!\n".format(i))
                print("Detect a smear! area = {}, circularity = {}".format(
                    area, circularity))
                cv2.drawContours(resI_smear_on_src, contours=[
                                 contour], contourIdx=-1, color=(255, 255, 0), thickness=2)
                cv2.drawContours(resI_smear_mask, contours=[
                                 contour], contourIdx=-1, color=(255, 255, 255), thickness=-1)

        return resI_smear_mask, resI_objects, resI_smear_on_src

    def find_longest_contour(self, contours):
        contour = contours[0]
        for i in range(1, len(contours)):
            if len(contours[i]) > len(contour):
                contour = contours[i]
        return contour

    def evaluate_circularity(self, contour):
        area_real = cv2.contourArea(contour)
        # Compute area_circle
        if 0:  # by min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area_circle = np.pi * radius**2
        else:  # by fitted ellipse
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            # (MA, ma) lengths of the major axis and minor axis
            area_circle = np.pi * MA * ma / 4
        circularity = area_circle / area_real
        return circularity

    def is_region_large_and_circular(self, contour):

        # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        area_real = cv2.contourArea(contour)
        circularity = self.evaluate_circularity(contour)
        print("area_real = {}, circularity = {}".format(area_real, circularity))

        # ! Modify THRESHOLD HERE !
        criteria_1 = 2000 > area_real > 100
        criteria_2 = abs(circularity - 1) < 2.0
        if 0:
            print(area_real, circularity)
        return (criteria_1 and criteria_2), area_real, circularity

    def do_morphological_open(self, img):
        kernel = np.ones((7, 7), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        return img

    def labels_to_color(self, labels):
        # convert labelled image to RGB color image
        # copied from: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        color_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # cvt to BGR for display
        color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR)
        color_img[label_hue == 0] = 0  # set bg label to black
        return color_img


if __name__ == "__main__":

    # Initialize
    folder_name = "cam_3/"
    image_reader = ImageReader(source="from_folder", folder_name=folder_name)

    imgs_list = collections.deque()
    motion_estimator = MotionEstimator()
    smear_detector = SmearDetector()

    # Read some images and saved to list
    MAX_IMAGE = 5000
    cnt_img = 0
    WINDOW_SIZE = 500
    while cnt_img < MAX_IMAGE and image_reader.is_next_image_available():

        # Read image
        src_img = image_reader.read(resize_rate=0.25)
        cnt_img += 1
        print("\nReading the {}th image".format(cnt_img))

        # Store image, and wait for filling the window
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        imgs_list.append(gray.astype(np.float))

        # Process
        if len(imgs_list) >= WINDOW_SIZE:
            if len(imgs_list) > WINDOW_SIZE:
                imgs_list.popleft()

            # Compute motion (static regions are whiter, which are smears)
            background_img = motion_estimator.calc_variation_image(imgs_list)

            # Detect smears
            resI_smear_mask, resI_objects, resI_smear_on_src = smear_detector.detect(
                src_img, background_img)

            # Display
            key_break, img_show = show_result_image(cnt_img,
                                                    [src_img, gray2color(background_img), resI_objects, resI_smear_on_src])

            # Write result image to file
            if 0:
                if "out_folder_bg" not in locals():
                    out_folder_bg = folder_name[:-1] + "_bg/"
                    create_folder(out_folder_bg)
                    out_folder_res = folder_name[:-1] + "_res/"
                    create_folder(out_folder_res)
                img_name = image_reader.get_prev_img_filename()
                cv2.imwrite(out_folder_bg + img_name, background_img)
                cv2.imwrite(out_folder_res + img_name, img_show)

            # break
            if key_break:
                break
