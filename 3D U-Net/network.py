import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H53DDataLoader
# from utils.img_utils import imsave
from utils import ops
from utils.Dense_Transformer_Networks_3D import *
from tifffile import imsave
import cv2
import re
import math


"""
This module builds a standard U-NET for semantic segmentation.
If want VAE using pixelDCL, please visit this code:
https://github.com/HongyangGao/UVAE
"""


class Unet_3D(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.savedir):
            os.makedirs(conf.savedir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        if self.conf.data_type == '3D':
            self.conv_size = (3, 3, 3)
            self.pool_size = (2, 2, 2)
            self.axis, self.channel_axis = (1, 2, 3), 4
            self.input_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width]
            if self.conf.add_dtn == True:
                # Operation in the list is trying to get the input shape of Dense transformer networks.
                self.dtn_input_shape = [self.conf.batch,int(self.conf.depth/(2**self.conf.dtn_location)),\
                                        int(self.conf.height/(2**self.conf.dtn_location)),
                                        int(self.conf.width/(2**self.conf.dtn_location)),
                                        self.conf.start_channel_num*(2**self.conf.dtn_location)]
                self.transform = DSN_Transformer_3D(self.dtn_input_shape,self.conf.control_points_ratio)
                self.insertdtn = self.conf.dtn_location
            else:
                self.insertdtn = -1

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(
            tf.int64, self.output_shape, name='annotations')
        self.predictions = self.inference(self.inputs)
        self.cal_loss()

    def cal_loss(self):
        one_hot_annotations = tf.one_hot(
            self.annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        losses = tf.losses.softmax_cross_entropy(
            one_hot_annotations, self.predictions, scope='loss/losses')

        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')

        self.decoded_predictions = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.annotations, self.decoded_predictions,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        self.softmax_predictions = tf.nn.softmax(self.predictions)




    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            if layer_index == self.insertdtn:
                outputs = self.build_down_block(outputs, name, down_outputs,first=is_first,TPS = True)
            else:
                outputs = self.build_down_block(outputs, name, down_outputs, first=is_first,TPS = False)  
            print("down ",layer_index," shape ", outputs.get_shape())          
        outputs = self.build_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            if layer_index == self.insertdtn:
                outputs = self.build_up_block(outputs, down_inputs, name,final=is_final,Decoder=True )
            else:
                outputs = self.build_up_block(outputs, down_inputs, name,final=is_final,Decoder=False )
            print("up ", layer_index, " shape ", outputs.get_shape())
        return outputs

    def build_down_block(self, inputs, name, down_outputs, first=False,TPS=False):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = ops.conv(inputs, out_num, self.conv_size,
                         name+'/conv1', self.conf.data_type)
        if TPS == True:
            conv1 = self.transform.Encoder(conv1, conv1)
        conv2 = ops.conv(conv1, out_num, self.conv_size,
                         name+'/conv2', self.conf.data_type)
        down_outputs.append(conv2)
        pool = ops.pool(conv2, self.pool_size, name +
                        '/pool', self.conf.data_type)
        return pool

    def build_bottom_block(self, inputs, name):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = ops.conv(
            inputs, 2*out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv2 = ops.conv(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        return conv2

    def build_up_block(self, inputs, down_inputs, name, final=False,Decoder=False):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, out_num, self.conv_size, name+'/conv1',
            self.conf.data_type, action=self.conf.action)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.conv_func()(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        if Decoder == True:
            conv2 = self.transform.Decoder(conv2,conv2)
        out_num = self.conf.class_num if final else out_num/2
        conv3 = ops.conv(
            conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type,
            not final)
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        data_reader = H53DDataLoader(self.conf.data_dir, self.conf.patch_size, self.conf.depth, self.conf.validation_id, self.conf.overlap_stepsize, self.conf.aug_flip, self.conf.aug_rotate)
        for train_step in range(1, self.conf.max_step+1):
            # if train_step % self.conf.test_interval == 0:
            #     inputs, annotations = data_reader.valid_next_batch()
            #     feed_dict = {self.inputs: inputs,
            #                  self.annotations: annotations}
            #     loss, summary = self.sess.run(
            #         [self.loss_op, self.valid_summary], feed_dict=feed_dict)
            #     self.save_summary(summary, train_step+self.conf.reload_step)
            #     print('----testing loss', loss)
            # el
            if train_step % self.conf.summary_interval == 0:
                inputs, annotations = data_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, train_step+self.conf.reload_step)
            else:
                inputs, annotations = data_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                predictions, loss, _ = self.sess.run(
                    [self.softmax_predictions, self.loss_op, self.train_op], feed_dict=feed_dict)
                print('----training loss', loss)
            if train_step % self.conf.save_interval == 0:#imsave('pre.tif',(predictions[0,:,:,:,1]*255).astype('uint8'))
                self.save(train_step+self.conf.reload_step)#imsave('gt.tif',(annotations[0]*255).astype('uint8'))

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        data_reader = H53DDataLoader(self.conf.data_dir, self.conf.patch_size, self.conf.validation_id, self.conf.overlap_stepsize)
        self.sess.run(tf.local_variables_initializer())
        # count = 0
        losses = []
        accuracies = []
        for i in range(data_reader.num_of_valid_patches):
            inputs, annotations, _ = data_reader.valid_next_batch()
            # if inputs.shape[0] < self.conf.batch:
            #     break

            # pseudo_inputs = np.zeros((1,32,32,32,2), dtype=np.float32)
            # pseudo_labels = np.zeros((1,32,32,32), dtype=np.float32)
            # CUT_MEAN = np.array((100.913811861, 121.187003401), dtype=np.float32)
            # pseudo_inputs -= CUT_MEAN

            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            loss, accuracy = self.sess.run([self.loss_op, self.accuracy_op],feed_dict=feed_dict)
            print('values----->', loss, accuracy)
            # count += 1
            losses.append(loss)
            accuracies.append(accuracy)
        print('Loss: ', np.mean(losses))
        print('Accuracy: ', np.mean(accuracies))

    def predict(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        data_reader = H53DDataLoader(self.conf.data_dir, self.conf.patch_size, self.conf.depth, self.conf.validation_id, self.conf.overlap_stepsize)
        self.sess.run(tf.local_variables_initializer())
        predictions = {}
        accs = []
        jacs = []
        tps = []
        fps = []
        fns = []
        tns = []
        for i in range(data_reader.num_of_valid_patches):
            inputs, annotations, location = data_reader.valid_next_batch()
            # if inputs.shape[0] < self.conf.batch:
            #     break
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            preds, acc, pred_mask = self.sess.run([self.softmax_predictions, self.accuracy_op,self.decoded_predictions], feed_dict=feed_dict)
            print('--->processing results for: ', location)
            print('accuracy:', acc)
            inters = np.logical_and(annotations, pred_mask)
            union = np.logical_or(annotations, pred_mask)
            jacs.append(np.sum(inters) / np.sum(union))
            accs.append(acc)
            tps.append(inters)
            fps.append(np.logical_and(pred_mask, np.logical_not(annotations)))
            fns.append(np.logical_and(annotations, np.logical_not(pred_mask)))
            tns.append(np.logical_and(np.logical_not(annotations), np.logical_not(pred_mask)))
        print('accuracy:',np.mean(accs))
        print('jaccard:',np.mean(jacs))
        print('average prec:', np.sum(tps) / (np.sum(tps) + np.sum(fps)))
        print('average reca:', np.sum(tps) / (np.sum(tps) + np.sum(fns)))
            # imsave('Z:\cochlea\\Unet_3D-master\model\\result\\image'+str(i).zfill(3)+'.tif', inputs[0,:,:,:,0].astype('uint8'))
            # imsave('Z:\cochlea\\Unet_3D-master2\model\\result2\\predict' + str(i).zfill(3) + '.tif',
            #        (preds[0, :, :, :, 0]*255).astype('uint8'))
            # imsave('Z:\cochlea\\Unet_3D-master2\model\\result2\\gt' + str(i).zfill(3) + '.tif',
            #        (annotations[0, :, :, :]).astype('uint8'))
            # for j in range(self.conf.depth):
            #     for k in range(self.conf.patch_size):
            #         for l in range(self.conf.patch_size):
            #             key = (location[0]+j, location[1]+k, location[2]+l)
            #             if key not in predictions.keys():
            #                 predictions[key] = []
            #             predictions[key].append(preds[0, j, k, l, :])
        # print('--->averaging results')
        # results = np.zeros((data_reader.t_d, data_reader.t_h, data_reader.t_w, self.conf.class_num), dtype=np.float32)
        # for key in predictions.keys():
        #     results[key[0], key[1], key[2]] = np.mean(predictions[key], axis=0)
        # print('--->saving results')
        # save_filename = 'results' + str(self.conf.test_step) + '_sub' + str(self.conf.validation_id) + '_overlap' + str(self.conf.overlap_stepsize) +'.npy'
        # save_file = os.path.join(self.conf.savedir, save_filename)
        # np.save(save_file, results)

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def predict_big_image(self):
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return

        scale = 0.5
        ImagePath = 'Y:\cochlea\Aligned_data\crop_cell2'
        MaskPath = './result_crop_cell2'


        if not os.path.exists(MaskPath):
            os.mkdir(MaskPath)
        files = os.listdir(ImagePath)
        files.sort(key=lambda i: int(re.match(r'(\d+)', i[29:]).group()))

        for start in range(0, len(files), 200):
            end = min(200+start, len(files))
            img = []

            files_cur = files[start:end]
            # files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
            for file in files_cur:
                print(file)
                img_temp = cv2.imread(os.path.join(ImagePath, file), cv2.CV_8UC1)
                img_temp = cv2.resize(img_temp,(int(img_temp.shape[1]*scale),int(img_temp.shape[0]*scale)),cv2.INTER_LINEAR)
                # if np.unique(img_temp).shape[0]>1:
                #     bug = 1
                img.append(img_temp)
                # img.append(img_temp)
            imgs = np.stack(img, axis=0)

            height = imgs.shape[1]
            width = imgs.shape[2]
            channel = imgs.shape[0]

            crop_height = self.conf.patch_size
            crop_width = self.conf.patch_size
            crop_channel = self.conf.depth
            step_xy = 384 #448
            step_z = 28
            i_count = math.ceil((height - crop_height) / step_xy) - 1
            j_count = math.ceil((width - crop_width) / step_xy) - 1
            t_count = math.ceil((channel - crop_channel) / step_z) - 1

            stitch_mask = np.zeros_like(imgs)
            for i in range(i_count + 2):
                if i == 2:
                    bug = 1
                if i < i_count + 1:
                    start_i = i * step_xy
                else:
                    start_i = height - crop_height
                for j in range(j_count + 2):
                    if j == 6:
                        bug = 2
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
                        crop_img = crop_img[np.newaxis,:,:,:,np.newaxis].astype('float32')
                        mean = np.mean(crop_img)
                        std = np.std(crop_img)
                        crop_img = (crop_img - mean) / std

                        feed_dict = {self.inputs: crop_img}
                        mask = self.sess.run(self.softmax_predictions, feed_dict=feed_dict)
                        mask = mask[0,:,:,:,1]>0.8
                        # cv2.imwrite('D:\keras\liuj\YY_mitochondria\\temp\\' + str(count).zfill(4) + '_mask.png', mask * 255)
                        # cv2.imwrite('D:\keras\liuj\YY_mitochondria\\temp\\' + str(count).zfill(4) + '_image.png', crop_img)
                        # count+=1
                        ###stitch
                        stitch_mask[start_z:start_z+crop_channel,start_i:start_i + crop_height, start_j:start_j + crop_width] = np.logical_or(mask,
                                                  stitch_mask[ start_z:start_z + crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width])
            ###save mask
            # stitch_mask = cv2.resize(stitch_mask, (width_ori, height_ori), interpolation=cv2.INTER_NEAREST)
            for tt in range(stitch_mask.shape[0]):
                cv2.imwrite(MaskPath + '\\' + str(tt+start).zfill(4)+'.png', (stitch_mask[tt] * 255).astype('uint8'))
