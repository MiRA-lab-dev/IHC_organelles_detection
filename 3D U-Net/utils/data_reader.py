import numpy as np
import h5py
import random
import os
from tifffile import imread,imsave
import math
import cv2
import re

class H53DDataLoader(object):

    def __init__(self, data_dir, patch_size, depth_size, valid_sub_id, overlap_stepsize, aug_flip=False, aug_rotate=False):
        data_files = []
        images_path = '/home/data/liuj/cochlea/CellBody segmentaion/Unet_3D-master/images'
        labels_path = '/home/data/liuj/cochlea/CellBody segmentaion/Unet_3D-master/labels'
        images = []
        labels = []
        # images_val = []
        # labels_val = []
        image_files = os.listdir(images_path)
        image_files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        label_files = os.listdir(labels_path)
        label_files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
            image = imread(os.path.join(images_path, image_file))
            label = imread(os.path.join(labels_path, label_file))
            image = cv2.resize(image.transpose([1, 2, 0]), (512, 512), cv2.INTER_LINEAR)
            label = cv2.resize(label.transpose([1, 2, 0]), (512, 512), cv2.INTER_NEAREST)
            image = np.transpose(image,[2,0,1])
            label = np.transpose(label,[2,0,1])
            label = label.astype('float32')/255.0
            mean = np.mean(image)
            std = np.std(image)
            image = (image-mean)/std
            # input_image = np.concatenate([np.expand_dims(image, axis=-1), seed], axis=-1)
            input_image = np.expand_dims(image, axis=-1)
            images.append(input_image)
            labels.append(label)
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)

        # images_val = np.stack(images_val, axis=0)
        # labels_val = np.stack(labels_val, axis=0)
        # data_files.append(h5py.File(os.path.join(data_dir, 'data.h5'), 'r'))
        
        # if aug_flip:
        #     data_files.append(h5py.File(os.path.join(data_dir, 'data_flip1.h5'), 'r'))
        #     data_files.append(h5py.File(os.path.join(data_dir, 'data_flip2.h5'), 'r'))
        #     data_files.append(h5py.File(os.path.join(data_dir, 'data_flip3.h5'), 'r')
        self.aug_rotate = aug_rotate


        # self.num_files = len(data_files)
        self.num_files = 1
        # inputs = [np.array(data_files[i]['X']) for i in range(self.num_files)]
        # labels = [np.array(data_files[i]['Y']) for i in range(self.num_files)]
        # inputs = [images, images_val]
        # labels = [labels, labels_val]
        inputs = [images]
        labels = [labels]

        self.t_n, self.t_d, self.t_h, self.t_w, self.t_c = inputs[0].shape
        self.d, self.h, self.w = depth_size, patch_size, patch_size

        # self.valid_sub_id = valid_sub_id - 1# leave-one-out cross validation 1-10
        # self.valid_sub_id = np.random.randint(0, self.t_n, 40)
        if not os.path.exists('index.npy'):
            self.valid_sub_id = np.random.randint(0, self.t_n, int(self.t_n*0.2))
            np.save('index.npy', self.valid_sub_id)
        else:
            self.valid_sub_id = np.load('index.npy')

        mask = np.ones(self.t_n, dtype=bool)
        mask[self.valid_sub_id] = False
        self.train_inputs = [inputs[i][mask] for i in range(self.num_files)]
        self.train_labels = [labels[i][mask] for i in range(self.num_files)]
        
        self.valid_inputs, self.valid_labels = inputs[0][self.valid_sub_id], labels[0][self.valid_sub_id]
        self.prepare_validation(overlap_stepsize)
        self.num_of_valid_patches = len(self.patches_ids)
        self.valid_patch_id = 0

    def next_batch(self, batch_size):
        batches_ids = set()
        while len(batches_ids) < batch_size:
            i = random.randint(0, self.num_files-1)
            # n = random.randint(0, self.t_n-2)
            n = random.randint(0, int(self.t_n*0.8))
            d = random.randint(0, self.t_d-self.d)
            h = random.randint(0, self.t_h-self.h)
            w = random.randint(0, self.t_w-self.w)
            if ((not self.aug_rotate) or (self.aug_rotate and i < self.num_files-6)):
                batches_ids.add((i, n, d, h, w))
            elif (i >= self.num_files-6 and i < self.num_files-4):
                batches_ids.add((i, n, h, d, w))
            elif (i >= self.num_files-4 and i < self.num_files-2):
                batches_ids.add((i, n, w, h, d))
            elif (i >= self.num_files-2):
                batches_ids.add((i, n, d, w, h))

        input_batches = []
        label_batches = []
        for i, n, d, h, w in batches_ids:
            images = self.train_inputs[i][n, d:d+self.d, h:h+self.h, w:w+self.w, :]
            labels = self.train_labels[i][n, d:d+self.d, h:h+self.h, w:w+self.w]
            rand_rotate = np.random.randint(0, 3)
            images = np.rot90(images, k=rand_rotate, axes=(1, 2))
            labels = np.rot90(labels, k=rand_rotate, axes=(1, 2))
            rand_flip = np.random.randint(0,4)
            if rand_flip == 1:
                images = np.fliplr(images)
                labels = np.fliplr(labels)
            elif rand_flip == 2:
                images = np.flipud(images)
                labels = np.flipud(labels)
            elif rand_flip == 3:
                images = images[:,:,::-1]
                labels = labels[:,:,::-1]


            input_batches.append(images)
            label_batches.append(labels)
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        return inputs, labels

    def prepare_validation(self, overlap_stepsize):
        self.patches_ids = []
        self.drange = list(range(0, self.t_d-self.d+1, overlap_stepsize))
        self.hrange = list(range(0, self.t_h-self.h+1, overlap_stepsize))
        self.wrange = list(range(0, self.t_w-self.w+1, overlap_stepsize))
        if (self.t_d-self.d) % overlap_stepsize != 0:
            self.drange.append(self.t_d-self.d)
        if (self.t_h-self.h) % overlap_stepsize != 0:
            self.hrange.append(self.t_h-self.h)
        if (self.t_w-self.w) % overlap_stepsize != 0:
            self.wrange.append(self.t_w-self.w)
        for n, _ in enumerate(self.valid_sub_id):
            for d in self.drange:
                for h in self.hrange:
                    for w in self.wrange:
                        self.patches_ids.append((n, d, h, w))
        
    def reset(self):
        self.valid_patch_id = 0

    def valid_next_batch(self):
        input_batches = []
        label_batches = []
        # self.num_of_valid_patches = len(self.patches_ids)
        batches_ids = set()
        while len(batches_ids) < 1:
            n, d, h, w = self.patches_ids[self.valid_patch_id]
            input_batches.append(self.valid_inputs[n, d:d+self.d, h:h+self.h, w:w+self.w, :])
            label_batches.append(self.valid_labels[n, d:d+self.d, h:h+self.h, w:w+self.w])
            batches_ids.add(self.valid_patch_id)
            self.valid_patch_id += 1
            if self.valid_patch_id == self.num_of_valid_patches:
                self.reset()
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        return inputs, labels, (d, h, w)
