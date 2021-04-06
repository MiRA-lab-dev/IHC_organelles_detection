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
from tifffile import imsave

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


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class MitochondriaDataset(utils.Dataset):

    def load_infos(self, count1, count2, imagepath, maskpath, ribbon_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("ultrastructure", 1, "mitochondria")

        for i, file in enumerate(os.listdir(ribbon_path)):
            # if os.path.exists(ribbon_path+file):
            self.add_image("ultrastructure", image_id=i, path=imagepath+file,
                           maskpath=ribbon_path+file)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ultrastructure":
            return info["ultrastructure"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = skimage.io.imread(info['maskpath'])
        mask = np.transpose(mask, axes=[1, 2, 0])
        # mask = mask[:, :] / 255
        # label = measure.label(mask, connectivity=2)
        labels = np.unique(mask)
        ################
        if os.path.exists(info['maskpath'].replace('ribbon', 'mem_ribbon')):
            mask_detection = skimage.io.imread(info['maskpath'].replace('ribbon', 'mem_ribbon'))
            mask_detection = np.transpose(mask_detection, axes=[1, 2, 0])
        else:
            mask_detection = mask
        ##################
        # newmask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], labels.shape[0]-1), dtype='int32')
        newmask = []
        mask_detections = []
        count = 0
        for i, label in enumerate(labels):
            if label == 0:
                continue
            temp = np.zeros(shape=(mask.shape[0], mask.shape[1], mask.shape[2]), dtype='uint8')
            temp[mask == label] = 1

            temp2 = np.zeros(shape=(mask.shape[0], mask.shape[1], mask.shape[2]), dtype='uint8')
            temp2[mask_detection==label] = 1
            if np.sum(temp2)==0:
                temp2 = temp

            # if np.unique(np.where(temp)[2]).shape[0]<3:
            #     continue
            # cv2.imshow('img',temp*255)
            # cv2.waitKey(0)
            # newmask[:, :, :, i] = temp
            if np.sum(temp2) < 500:
                continue
            # print('sum:', np.sum(temp))
            newmask.append(temp)
            mask_detections.append(temp2)
            count = count + 1

        if len(newmask) > 0:
            newmask = np.stack(newmask, axis=3)
            mask_detections = np.stack(mask_detections, axis=3)
        else:
            newmask = np.zeros(shape=(mask.shape[0], mask.shape[1], mask.shape[2], 1), dtype='uint8')
            mask_detections = np.zeros(shape=(mask.shape[0], mask.shape[1], mask.shape[2], 1), dtype='uint8')
        # rgb = color.label2rgb(label)
        # cv2.imshow('label',np.reshape(rgb,(768,1024,3)))
        # cv2.waitKey(0)
        assert newmask.shape[3]==mask_detections.shape[3]
        # class_ids = 1
        class_ids = np.ones(count, dtype='int32')
        return newmask, mask_detections, class_ids

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_mask")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

config = MitochondriaConfig()
config.display()

dataset_train = MitochondriaDataset()
dataset_train.load_infos(0, 6300, '/home/data/liuj/cochlea/ribbon_detection/images512/', './mem_ribbon/train/masks/','./ribbon/train/masks/')
dataset_train.prepare()

# Validation dataset
dataset_val = MitochondriaDataset()
dataset_val.load_infos(168, 210, '/home/data/liuj/cochlea/ribbon_detection/images512/', './mem_ribbon/val/masks/','./ribbon/val/masks/')
dataset_val.prepare()

# image_ids = np.random.choice(dataset_train.image_ids, 20)
# for image_id in range(6, 73):
#     # image = dataset_train.load_image(image_id)
#     # mask, class_ids = dataset_train.load_mask(image_id)
#     original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#         modellib.load_image_gt(dataset_val, config,
#                                image_id, augment=False, use_mini_mask=False)
#     # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#     #                             dataset_val.class_names, figsize=(8, 8))
#     images_color = visualize.display_3D_boxes(original_image, gt_bbox, gt_mask, norm=False)
#     # imsave('.\gt\\' + str(image_id).zfill(3) + '.tif', images_color)
#     for m in range(32):
#         cv2.imwrite('.\gt\\'+str(image_id).zfill(3)+'_'+str(m).zfill(2)+'.png',images_color[m])
    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 仅使用第3块GPU, 从0开始
# model = modellib.MaskRCNN(mode="training", config=config,
#                           model_dir=MODEL_DIR)
# model_path = 'Y:\cochlea\\ribbon_detection\MaskRCNN-3D\logs\mitochondria20200422T1101\mask_rcnn_mitochondria_0043.h5'
# model.load_weights(model_path, by_name=True)

# # #
# init_with = "last"  # imagenet, coco, or last
#
# if init_with == "imagenet":
#     model.load_weights(model.get_imagenet_weights(), by_name=True)
# elif init_with == "coco":
#     # Load weights trained on MS COCO, but skip layers that
#     # are different due to the different number of classes
#     # See README for instructions to download the COCO weights
#     model.load_weights(COCO_MODEL_PATH, by_name=True,
#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                                 "mrcnn_bbox", "mrcnn_mask"])
# elif init_with == "last":
#     # Load the last model you trained and continue training
#     model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=5,
#             layers='heads')
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=30,
#             layers="all")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=20,
#             layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

class InferenceConfig(MitochondriaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()[1]
model_path = './logs_mask/mitochondria20200607T0756/mask_rcnn_mitochondria_0010.h5'
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# image_ids = np.random.choice(dataset_val.image_ids,10)
match_number = 0
gt_number = 0
detect_number = 0
missed = []
tps = []
fps = []
fns = []
tns = []
for image_id in range(73):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    # log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_bbox)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset_val.class_names, figsize=(8, 8))
    # plt.savefig('.\predict\gt\\'+str(image_id).zfill(3)+'.png', format='png')

    results = model.detect([original_image], verbose=1)
    r = results[0]
    index = r['rpn_scores'][0]>0.5
    images_color = visualize.display_3D_boxes(original_image, r['rois'], r['masks'], norm=False)
    # imsave('.\predict\\'+ str(image_id).zfill(3) +'mask.tif', (r['masks'][:,:,:,0].transpose(2,0,1)*255).astype('uint8'))
    # imsave('.\predict\\' + str(image_id).zfill(3) +'.tif', images_color.transpose([0,3,1,2]))
    # for m in range(32):
    #     cv2.imwrite('.\predict\\'+str(image_id).zfill(3)+'_'+str(m).zfill(2)+'.png',images_color[m])

    # predicted_boxes = r['rpn_rois'][0][index]*np.array([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL])
    predicted_boxes = r['rois']
    overlaps = utils.compute_overlaps(predicted_boxes, gt_bbox)
    if len(predicted_boxes) == 0:
        if gt_class_id.shape[0]!=0:
            missed.append(image_id)
        gt_number += gt_class_id.shape[0]
        continue
    iou_max = np.max(overlaps, axis=0)
    # iou_argmax = np.argmax(overlaps, axis=0)
    positive_ids = np.where(iou_max >= 0.23)[0]
    # matched_gt_boxes = iou_argmax[positive_ids]
    match_number += len(positive_ids)
    gt_number += gt_class_id.shape[0]
    if len(set(positive_ids))<gt_class_id.shape[0]:
        missed.append(image_id)
    detect_number += predicted_boxes.shape[0]

    masks = r['masks']
    mask = np.zeros(shape=(512, 512, 32), dtype='uint8')
    if masks.shape[0] == 512:
        for t in range(masks.shape[3]):
            mask = np.logical_or(mask, masks[:, :, :, t])
    # cv2.imwrite('.\predict\\' + str(image_id).zfill(3) + '.png',mask*255)

    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset_val.class_names, r['scores'], ax=get_ax())
    # plt.savefig('.\predict\\' + str(image_id).zfill(3) + '.png', format='png')
    # plt.close()
    gt_mask = gt_mask[:,:,:,0]
    tps.append(np.logical_and(gt_mask, mask))
    fps.append(np.logical_and(mask, np.logical_not(gt_mask)))
    fns.append(np.logical_and(gt_mask, np.logical_not(mask)))
    tns.append(np.logical_and(np.logical_not(gt_mask), np.logical_not(mask)))
print('average prec:', np.sum(tps)/(np.sum(tps)+np.sum(fps)))
print('average reca:', np.sum(tps) / (np.sum(tps) + np.sum(fns)))
# print('recall:', match_number/gt_number)
# print('precision:', match_number/detect_number)
# print('missed:', missed)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)

print("mAP: ", np.mean(APs))
