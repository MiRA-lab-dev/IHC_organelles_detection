import os
import cv2
from skimage.morphology import remove_small_objects, erosion,disk, opening, closing
import numpy as np
from skimage.morphology import watershed,label
from skimage.color import label2rgb,lab2rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from tifffile import imsave,imread
import mahotas as mh


if __name__ == '__main__':
    path = 'result_crop_cell2'
    files = os.listdir(path)
    MaskPath = './result_crop_cell2_ws_rgb'
    wsPath = './result_crop_cell2_ws'

    if not os.path.exists(MaskPath):
        os.mkdir(MaskPath)

    for file in files:
        if int(file[:4])<438:
            continue
        image = cv2.imread(os.path.join(path, file), cv2.CV_8UC1)
        # image = opening(image, selem=disk(10))
        distance = ndi.distance_transform_edt(image)

        local_maxi = peak_local_max(distance, labels=image, footprint=np.ones((80, 80)),  indices=False)
        markers = ndi.label(local_maxi)[0]

        labels = watershed(-distance, markers, mask=image)
        imsave(wsPath + '\\' + file, labels)
        # border = mh.labeled.borders(labels)
        # labels[border>0] = 0
        rgb = label2rgb(labels)
        imsave(MaskPath + '\\' + file, (rgb[:, :, ::-1] * 255).astype('uint8'))


    for file in os.listdir(MaskPath):
        if int(file[:4]) < 306 or int(file[:4]) > 340:
            continue
        label = imread(os.path.join(MaskPath, file))
        rgb = label2rgb(label)
        # imsave('./result_crop_ws_rgb/'+file,rgb)
        cv2.imwrite('./result_crop_cell2_ws_rgb/' + file, (rgb * 255).astype('uint8'))
