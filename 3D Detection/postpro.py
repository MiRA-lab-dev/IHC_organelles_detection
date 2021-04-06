import cv2
import os
from skimage.measure import regionprops
from scipy import ndimage as ndi
# ndi.distance_transform_edt()
import re
import numpy as np


if __name__ == '__main__':
    ribbon_path = 'Y:\cochlea\\ribbon_detection\MaskRCNN-3D\\ribbon_crop_cell2_final\\all'
    cell_path = 'Y:\cochlea\Aligned_data\crop_cell2_fiji\\trakem2.1591008294190.2101327853.3322066\proofreading'
    save_path = 'Y:\cochlea\\ribbon_detection\MaskRCNN-3D\\ribbon_crop_cell2_final\postpro2'
    ribbon_files = os.listdir(ribbon_path)
    ribbon_files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    cell_files = os.listdir(cell_path)
    cell_files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for i, file in enumerate(ribbon_files):
        # if i!=952:
        #     continue
        ribbon = cv2.imread(os.path.join(ribbon_path, file), cv2.CV_16UC1)
        cell = cv2.imread(os.path.join(cell_path, cell_files[i]), cv2.CV_16UC1)
        cell_mask = (cell>0).astype('uint16')
        props = regionprops(cell_mask)
        (min_row, min_col, max_row, max_col) = props[0].bbox
        # col = int(min_col/3+max_col/3*2)
        col = int(min_col / 2 + max_col / 2)
        ribbon = ribbon*cell_mask
        ribbon[:,col:] = 0
        cv2.imwrite(os.path.join(save_path, str(i).zfill(4)+'.png'), ribbon)

