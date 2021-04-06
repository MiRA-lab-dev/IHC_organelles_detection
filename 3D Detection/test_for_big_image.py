import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
import skimage
from skimage import measure,color
from skimage.measure import regionprops
from tifffile import imsave
from pairwise_match_lj import pair_match

####线粒体配置
class MitochondriaConfig(Config):

    # Give the configuration a recognizable name
    NAME = "mitochondria"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + mitochondria

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512 #height
    IMAGE_MAX_DIM = 512 #width

    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    IMAGE_CHANNEL = 32

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]


    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 512
    TRAIN_ROIS_PER_IMAGE = 256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 32

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  # 256

    # RPN_NMS_THRESHOLD = 0.5

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 76


class InferenceConfig(MitochondriaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_mask")

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# model_path = model.find_last()[1]
model_path = 'Y:\cochlea\\ribbon_detection\MaskRCNN-3D\logs_mask\mitochondria20200607T0756\mask_rcnn_mitochondria_0010.h5'
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

if __name__ == '__main__':
    def merge(box1,box2):
        (min_row1, min_col1, max_row1, max_col1) = box1
        (min_row2, min_col2, max_row2, max_col2) = box2
        min_row = min(min_row1, min_row2)
        min_col = min(min_col1, min_col2)
        max_row = max(max_row1, max_row2)
        max_col = max(max_col1, max_col2)
        return (min_row, min_col, max_row, max_col)

    scale = 1
    ImagePath = 'Y:\cochlea\Aligned_data\crop_cell2'
    MaskPath = './ribbon_crop_cell2_binary'
    cell_path = 'Y:\cochlea\Aligned_data\crop_cell2_fiji\\trakem2.1591008294190.2101327853.3322066\proofreading'

    if not os.path.exists(MaskPath):
        os.mkdir(MaskPath)


    files = os.listdir(ImagePath)
    files.sort(key=lambda i: int(re.match(r'(\d+)', i[29:]).group()))
    cell_files = os.listdir(cell_path)
    cell_files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    length = 64

    for start in range(0, len(files), length):
        end = min(length+start, len(files))
        img = []
        cells = []

        files_cur = files[start:end]
        cell_files_cur = cell_files[start:end]
        dict_box = {}

        for file, cell_file in zip(files_cur, cell_files_cur):
            img_temp = cv2.imread(os.path.join(ImagePath, file), cv2.CV_8UC1)
            img_temp = cv2.resize(img_temp, (int(img_temp.shape[1]*scale),int(img_temp.shape[0]*scale)),cv2.INTER_LINEAR)
            img.append(img_temp)
            # if np.unique(img_temp).shape[0]>1:
            # img_temp = img_temp[1500:,1500:]
            #     bug = 1

            # cell_temp = cv2.imread(os.path.join(cell_path, cell_file), cv2.CV_16UC1)
            # cell_temp = cv2.resize(cell_temp, (int(cell_temp.shape[1]*scale),int(cell_temp.shape[0]*scale)),cv2.INTER_NEAREST)
            # props = regionprops(cell_temp)
            # for p in props:
            #     l = p.label
            #     (min_row, min_col, max_row, max_col) = p.bbox
            #     if l not in dict_box:
            #         dict_box[l] = p.bbox
            #     else:
            #         pre_box = dict_box[l]
            #         box = merge(pre_box, p.bbox)
            #         dict_box[l] = box
            #
            # cells.append(cell_temp)

        imgs = np.stack(img, axis=0)
        # cells = np.stack(cells, axis=0)

        height = imgs.shape[1]
        width = imgs.shape[2]
        channel = imgs.shape[0]

        crop_height = inference_config.IMAGE_HEIGHT
        crop_width = inference_config.IMAGE_WIDTH
        crop_channel = inference_config.IMAGE_CHANNEL
        step_xy = 480
        step_z = 28
        i_count = math.ceil((height - crop_height) / step_xy) - 1
        j_count = math.ceil((width - crop_width) / step_xy) - 1
        t_count = math.ceil((channel - crop_channel) / step_z) - 1
        count = 0

        stitch_mask = np.zeros(list(imgs.shape), dtype='uint16')
        for i in range(i_count + 2):
            if i < i_count + 1:
                start_i = i * step_xy
            else:
                start_i = height - crop_height
            for j in range(j_count + 2):
                if j < j_count + 1:
                    start_j = j * step_xy
                else:
                    start_j = width - crop_width
                for t in range(t_count + 2):
                    if t < t_count + 1:
                        start_z = t * step_z
                        crop_img = imgs[t * step_z:t * step_z + crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width]
                    else:
                        start_z = channel - crop_channel
                        crop_img = imgs[channel - crop_channel:channel, start_i:start_i + crop_height, start_j:start_j + crop_width]
                    # imsave('image'+str(count).zfill(3)+'.tif', crop_img)
                    # count += 1
                    crop_img = np.transpose(crop_img,[1,2,0])[:, :, :, np.newaxis].astype('float32')
                    # if len(crop_img.shape) != 4:
                    #     crop_img = np.stack([crop_img, crop_img, crop_img], axis=3)

                    results = model.detect([crop_img], verbose=1)
                    r = results[0]
                    masks = r['masks']
                    mask = np.zeros(shape=(crop_height, crop_width, crop_channel), dtype='uint8')

                    if masks.shape[0] == crop_height:
                        # cv2.imwrite('.\predict\\' + str(image_id).zfill(4) + '.png', mask * 255)
                        # continue
                        for t in range(masks.shape[3]):
                            # mask += masks[:, :, t].astype('uint8')
                            mask = np.logical_or(mask, masks[:, :, :, t])
                            # mask[masks[:,:,:,t]>0] = label_start
                            # label_start+=1


                    # mask = mask[0,:,:,:,1]>0.8
                    # prob = np.transpose(mask,[2,0,1])*255
                    # cur_mask = stitch_mask[start_z:start_z + crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width]
                    # stitch_mask[start_z:start_z+crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width] = \
                    #     np.where(cur_mask>0,
                    #              (cur_mask+prob)/2.,
                    #              prob)
                    # imsave('predict'+str(count).zfill(3)+'.tif',(mask*255).astype('uint8'))
                    # cv2.imwrite('D:\keras\liuj\YY_mitochondria\\temp\\' + str(count).zfill(4) + '_mask.png', mask * 255)
                    # cv2.imwrite('D:\keras\liuj\YY_mitochondria\\temp\\' + str(count).zfill(4) + '_image.png', crop_img)
                    # count+=1
                    ###stitch
                    mask = np.transpose(mask, [2, 0, 1])
                    stitch_mask[start_z:start_z+crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width] = np.logical_or(mask,
                                              stitch_mask[start_z:start_z + crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width])

        ###save mask
        # stitch_mask = cv2.resize(stitch_mask, (width_ori, height_ori), interpolation=cv2.INTER_NEAREST)
        for tt in range(stitch_mask.shape[0]):
            cv2.imwrite(MaskPath + '\\' + str(start+tt).zfill(4)+'.png', (stitch_mask[tt]).astype('uint8'))