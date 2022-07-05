import sys
import os
import glob
import cv2 as cv
from features import find_keypoint
from stitching import stitch_img

def load_data(foldername):
    # Load img
    filename_lst = sorted(glob.glob(os.path.join(foldername, '*.JPG')))
    img_lst = []
    for filename in filename_lst:
        img_lst.append(cv.imread(filename))

    # Load focal length
    focal_length_file = open(os.path.join(foldername, 'focalLength.txt'))
    focal_length_lst = []
    for i in range(len(img_lst)):
        focal_length_lst.append(float(focal_length_file.readline()) * 8)
    
    return img_lst, focal_length_lst
    

if __name__ == "__main__":
    foldername = sys.argv[1]
    img_lst, focal_length_lst = load_data(foldername)
    
    matched_keypoint = find_keypoint(img_lst)
    stitched_result = stitch_img(img_lst, focal_length_lst, matched_keypoint)
    
    # Save result image
    cv.imwrite('result.png', stitched_result.astype('uint8'))