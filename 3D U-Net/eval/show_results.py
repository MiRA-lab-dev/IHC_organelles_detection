import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

results_filename = 'Z:\cochlea\\Unet_3D-master\model\\result\\results44000_sub10_overlap16.npy'
# data_filename = '/tempspace/zwang6/infant_brain/h5_data/data.h5'
valid_sub_id = 10

param = 60

# load results
results = np.load(results_filename)
pred_labels = np.argmax(results, axis=3)

# load labels
# data_file = h5py.File(data_filename, 'r')
# inputs, labels = np.array(data_file['X']), np.array(data_file['Y'])
# valid_inputs, valid_labels = inputs[valid_sub_id-1], labels[valid_sub_id-1]
valid_inputs = np.load('Z:\cochlea\\Unet_3D-master\model\\result\\val_images.npy')
for i in range(16):
    cv2.imwrite(str(i)+'pred.png', (pred_labels[i, :, :]*255).astype('uint8'))
# for i in range(16):
#     cv2.imwrite(str(i)+'valid.png', (valid_inputs[i,:,:,0]).astype('uint8'))

# show
fig = plt.figure()
a=fig.add_subplot(1,2,1)
pred_img = pred_labels[:,:,param]
imgplot = plt.imshow(pred_img)
a.set_title('Pred')
a=fig.add_subplot(1,2,2)
true_img = valid_labels[:,:,param]
imgplot = plt.imshow(true_img)
a.set_title('Truth')

plt.show()