import os
import numpy as np
import cv2
from pairwise_match_lj import pair_match


if __name__ == '__main__':
    path = 'Y:\cochlea\\ribbon_detection\MaskRCNN-3D\\ribbon_crop_cell2_final'
    folders = os.listdir(path)
    overlap = 4
    length = 16
    for i in range(0, len(folders)-1):
        files = os.listdir(os.path.join(path, folders[i]))
        images1 = []
        start = int(files[-length][:4])
        for file in files[-length:]:
            image = cv2.imread(os.path.join(path, folders[i], file),cv2.CV_16UC1)
            images1.append(image)
        images1 = np.stack(images1, axis=0)

        files = os.listdir(os.path.join(path, folders[i+1]))
        images2 = []
        for file in files[:length]:
            image = cv2.imread(os.path.join(path, folders[i+1], file),cv2.CV_16UC1)
            images2.append(image)
        images2 = np.stack(images2, axis=0)

        out = pair_match(images1, images2, direction=1, halo_size=overlap)
        for mm in range(out.shape[0]):
            cv2.imwrite('Y:\cochlea\\ribbon_detection\MaskRCNN-3D\\ribbon_crop_cell2\\'+str(start).zfill(4)+'.png',out[mm])
            start+=1




