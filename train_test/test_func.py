"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import math
import numpy as np
import tensorflow as tf

def extract_roi_from_nd_volume(volume, roi_center, roi_shape, fill = 'random'):
    '''Extract an roi from a nD volume
        volume      : input nD numpy array
        roi_center  : center of roi with position
        output_shape: shape of roi
        '''
    input_shape = volume.shape
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = roi_shape)
    else:
        output = np.zeros(roi_shape)
    r0max = [int(x/2) for x in roi_shape]
    r1max = [roi_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - roi_center[i]) for i in range(len(r0max))]
    out_center = r0max
    if(len(roi_center)==3):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                    range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                    range(roi_center[2] - r0[2], roi_center[2] + r1[2]))]
    elif(len(roi_center)==4):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]),
                      range(out_center[3] - r0[3], out_center[3] + r1[3]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                      range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                      range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                      range(roi_center[3] - r0[3], roi_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output

def set_roi_to_nd_volume(volume, roi_center, sub_volume):
    '''Set an roi of an ND volume with a sub volume
        volume: an ND numpy array
        roi_center: center of roi
        sub_volume: the sub volume that will be copied
        '''
    volume_shape = volume.shape
    patch_shape  = sub_volume.shape
    output_volume = volume
    for i in range(len(roi_center)):
        if(roi_center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - roi_center[i]) for i in range(len(r0max))]
    patch_center = r0max


    if(len(roi_center) == 3):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(roi_center) == 4):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                             range(roi_center[3] - r0[3], roi_center[3] + r1[3]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                          range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume

def volume_probability_prediction_3d_roi(img, data_shape, label_shape,
                                         class_num, batch_size, sess, x, proby):
    '''
        Test one image with sub regions along x, y, z axis
        img        : a 4D numpy array with shape [D, H, W, C]
        data_shape : input 4d tensor shape
        label_shape: output 4d tensor shape
        class_num  : number of output class
        batch_size : batch size for testing
        sess       : tensorflow session that can run a graph
        x          : input tensor of the graph
        proby      : output tensor of the graph
        '''
    [D, H, W, C] = img.shape
    prob = np.zeros([D, H, W, class_num], np.float32)
    sub_image_patches = []
    sub_image_centers = []
    roid_half0 = int(label_shape[0]/2); roid_half1 = label_shape[0] - roid_half0
    roih_half0 = int(label_shape[1]/2); roih_half1 = label_shape[1] - roih_half0
    roiw_half0 = int(label_shape[2]/2); roiw_half1 = label_shape[2] - roiw_half0
    
    # get image patches
    for centerd in range(roid_half0, D + roid_half0, label_shape[0]):
        centerd = min(centerd, D - roid_half1)
        for centerh in range(roih_half0, H + roih_half0, label_shape[1]):
            centerh =  min(centerh, H - roih_half1)
            for centerw in range(roiw_half0, W + roiw_half0, label_shape[2]):
                centerw =  min(centerw, W - roiw_half1)
                roi_center = [centerd, centerh, centerw, int(C/2)]
                sub_image_centers.append(roi_center)
                sub_image = extract_roi_from_nd_volume(img, roi_center, data_shape, fill = 'random')
                sub_image_patches.append(sub_image)

    # inference with image patches
    total_batch = len(sub_image_patches)
    print("total batch number and input size", total_batch, img.shape)
    max_mini_batch = int((total_batch + batch_size -1)/batch_size)
    for mini_batch_idx in range(max_mini_batch):
        batch_end_idx = min((mini_batch_idx+1)*batch_size, total_batch)
        batch_start_idx = batch_end_idx - batch_size
        data_mini_batch = sub_image_patches[batch_start_idx:batch_end_idx]
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        prob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(batch_start_idx, batch_end_idx):
            roi_center = sub_image_centers[batch_idx]
            roi_center[3] = int(class_num/2)
            prob = set_roi_to_nd_volume(prob, roi_center, prob_mini_batch[batch_idx-batch_start_idx])
    return prob


def get_rotation_augmented_prediction(img, data_shape, label_shape, class_num, batch_size, sess, x, proby, lab, aug_number):
    [D, H, W, C] = img.shape
    lab = np.reshape(lab, [D, H, W])
    pred_list = []
    dice_list = []
    for i in range(aug_number):
#        pred = volume_probability_prediction_3d_roi(img, data_shape, label_shape,
#                                                   class_num, batch_size, sess, x, proby)
        angle = random.random()*360
        img_rotate  = ndimage.interpolation.rotate(img,angle, axes = (1, 2), reshape = False)
        pred_rotate = volume_probability_prediction_3d_roi(img_rotate, data_shape, label_shape,
                                                   class_num, batch_size, sess, x, proby)
        pred = ndimage.interpolation.rotate(pred_rotate, - angle, axes = (1, 2),reshape = False)
        pred = np.asarray(np.argmax(pred, axis = 3), np.float32)
        dice = binary_dice(lab, pred)
        pred_list.append(pred)
        dice_list.append(dice)
    pred = np.asarray(pred_list)
    # the following way to calculate probability is only valid for binary segmentation
    # get foreground probability
    pred_p = np.mean(pred, axis = 0)
    return pred_p, dice_list

def get_dropout_augmented_prediction(img, data_shape, label_shape, class_num, batch_size, sess, x, proby, lab, aug_number):
    [D, H, W, C] = img.shape
    lab = np.reshape(lab, [D, H, W])
    pred_list = []
    dice_list = []
    for i in range(aug_number):
        pred = volume_probability_prediction_3d_roi(img, data_shape, label_shape,
                                                   class_num, batch_size, sess, x, proby)
        pred = np.asarray(np.argmax(pred, axis = 3), np.float32)
        dice = binary_dice(lab, pred)
        pred_list.append(pred)
        dice_list.append(dice)
    pred = np.asarray(pred_list)
    # the following way to calculate probability and label is only valid for binary segmentation
    # get foreground probability
    pred_p = np.mean(pred, axis = 0)
    return pred_p, dice_list


def get_flip_augmented_prediction(img, data_shape, label_shape,
                           class_num, batch_size, sess, x, proby, lab):
    [D, H, W, C] = img.shape
    lab = np.reshape(lab, [D, H, W])
    flip1 = np.flip(img, axis = 2)
    flip2 = np.flip(img, axis = 1)
    flip3 = np.flip(flip1, axis = 1)
    
    img_trans = np.transpose(img, axes = [0, 2, 1, 3])
    flip4 = np.flip(img_trans, axis = 2)
    flip5 = np.flip(img_trans, axis = 1)
    flip6 = np.flip(flip4, axis = 1)
    
    all_input  = np.concatenate((img, flip1, flip2, flip3, img_trans, flip4, flip5, flip6))
    all_output = volume_probability_prediction_3d_roi(all_input, data_shape, label_shape,
                                               class_num, batch_size, sess, x, proby)
    all_output = np.asarray(np.argmax(all_output, axis = 3), np.float32)
    outp1  = all_output[0:D]
    outp2 = np.flip(all_output[D:2*D], axis = 2)
    outp3 = np.flip(all_output[2*D:3*D], axis = 1)
    outp4 = np.flip(all_output[3*D:4*D], axis = 1)
    outp4 = np.flip(outp4, axis = 2)
    
    outp5 = all_output[4*D:5*D]
    outp6 = np.flip(all_output[5*D:6*D], axis = 2)
    outp7 = np.flip(all_output[6*D:7*D], axis = 1)
    outp8 = np.flip(all_output[7*D:8*D], axis = 1)
    outp8 = np.flip(outp8, axis = 2)
    
    pred = np.asarray([outp1, outp2, outp3, outpt4, outp5, outp6, outp7, outp8])
    dice1 = binary_dice(lab, outp1); dice2 = binary_dice(lab, outp2)
    dice3 = binary_dice(lab, outp1); dice4 = binary_dice(lab, outp2)
    dice5 = binary_dice(lab, outp1); dice6 = binary_dice(lab, outp2)
    dice7 = binary_dice(lab, outp1); dice8 = binary_dice(lab, outp2)
    dice_list = [dice1, dice2, dice3, dice4, dice5, dice6, dice7, dice8]
    # get foreground probability
    pred_p = np.mean(pred, axis = 0)
    return pred_p, dice_list