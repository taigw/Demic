# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage
from Demic.util.image_process import *
from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import *


# Dice evaluation
def binary_dice(s, g, resize = False):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    resize: if s and g have different shapes, resize s to match g.
    """
    assert(len(s.shape)== len(g.shape))
    if(resize):
        size_match = True
        for i in range(len(s.shape)):
            if(s.shape[i] != g.shape[i]):
                size_match = False
                break
        if(size_match is False):
            s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = 2.0*s0/(s1 + s2 + 0.00001)
    return dice

def dice_of_images(s_name, g_name):
    s = load_image_as_4d_array(s_name)['data_array']
    g = load_image_as_4d_array(g_name)['data_array']
    dice = binary_dice(s, g)
    return dice

# IOU evaluation
def binary_iou(s,g):
    assert(len(s.shape)== len(g.shape))
    intersecion = np.multiply(s, g)
    union = np.asarray(s + g >0, np.float32)
    iou = intersecion.sum()/(union.sum() + 1e-10)
    return iou

def iou_of_images(s_name, g_name):
    s = load_image_as_4d_array(s_name)['data_array']
    g = load_image_as_4d_array(g_name)['data_array']
    margin = (3, 8, 8)
    g = get_detection_binary_bounding_box(g, margin)
    return binary_iou(s, g)

# Hausdorff evaluation
def slice_to_contour(img):
    assert(len(img.shape) == 2)
    point_list = []
    [H, W] = img.shape
    offset_h  = [ -1, 1,  0, 0]
    offset_w  = [ 0, 0, -1, 1]
    for h in range(1, H-1):
        for w in range(1, W-1):
            if(img[h, w] > 0):
                edge_flag = False
                for idx in range(4):
                    if(img[h + offset_h[idx], w + offset_w[idx]] == 0):
                        edge_flag = True
                        break
                if(edge_flag):
                    point_list.append([h, w])
    return point_list

def volume_to_surface(img):
    strt = ndimage.generate_binary_structure(3,2)
    img  = ndimage.morphology.binary_closing(img, strt, 5)
    point_list = []
    [D, H, W] = img.shape
    offset_d  = [-1, 1,  0, 0,  0, 0]
    offset_h  = [ 0, 0, -1, 1,  0, 0]
    offset_w  = [ 0, 0,  0, 0, -1, 1]
    for d in range(1, D-1):
        for h in range(1, H-1):
            for w in range(1, W-1):
                if(img[d, h, w] > 0):
                    edge_flag = False
                    for idx in range(6):
                        if(img[d + offset_d[idx], h + offset_h[idx], w + offset_w[idx]] == 0):
                            edge_flag = True
                            break
                    if(edge_flag):
                        point_list.append([d, h, w])
    return point_list

def hausdorff_distance_from_one_surface_to_another(point_list_s, point_list_g, spacing):
    dis_square = 0.0
    n_max = 300
    if(len(point_list_s) > n_max):
        point_list_s = random.sample(point_list_s, n_max)
    for ps in point_list_s:
        ps_nearest = 1e10
        for pg in point_list_g:
            dd = spacing[0]*(ps[0] - pg[0])
            dh = spacing[1]*(ps[1] - pg[1])
            dw = spacing[2]*(ps[2] - pg[2])
            temp_dis_square = dd*dd + dh*dh + dw*dw
            if(temp_dis_square < ps_nearest):
                ps_nearest = temp_dis_square
        if(dis_square < ps_nearest):
            dis_square = ps_nearest
    return math.sqrt(dis_square)

def hausdorff_distance_from_one_contour_to_another(point_list_s, point_list_g, spacing=[1.0, 1.0]):
    dis_square = 0.0
    n_max = 300
    if(len(point_list_s) > n_max):
        point_list_s = random.sample(point_list_s, n_max)
    for ps in point_list_s:
        ps_nearest = 1e8
        for pg in point_list_g:
            dh = spacing[0]*(ps[0] - pg[0])
            dw = spacing[1]*(ps[1] - pg[1])
            temp_dis_square = dh*dh + dw*dw
            if(temp_dis_square < ps_nearest):
                ps_nearest = temp_dis_square
        if(dis_square < ps_nearest):
            dis_square = ps_nearest
    dis = math.sqrt(dis_square)
    if(dis == 1e4):
        dis = 50
    return dis

def binary_hausdorff2d(s, g, spacing = [1.0, 1.0]):
    if(len(s.shape) == 4):
        s = np.reshape(s, s.shape[1:3])
        g = np.reshape(g, g.shape[1:3])
    else:
        raise ValueError("invalid volume dimention {0:}".format(len(s.shape)))
    point_list_s = slice_to_contour(s)
    point_list_g = slice_to_contour(g)
    dis1 = hausdorff_distance_from_one_contour_to_another(point_list_s, point_list_g, spacing)
    dis2 = hausdorff_distance_from_one_contour_to_another(point_list_g, point_list_s, spacing)
    return max(dis1, dis2)

def binary_hausdorff3d(s, g, spacing):
    if(len(s.shape) == 4):
        s = np.reshape(s, s.shape[:3])
        g = np.reshape(g, g.shape[:3])
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    if(not(Ds==Dg and Hs==Hg and Ws==Wg)):
        s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    scale = [1.0, spacing[1], spacing[2]]
    s_resample = ndimage.interpolation.zoom(s, scale, order = 0)
    g_resample = ndimage.interpolation.zoom(g, scale, order = 0)
    point_list_s = volume_to_surface(s_resample)
    point_list_g = volume_to_surface(g_resample)
    new_spacing = [spacing[0], 1.0, 1.0]
    dis1 = hausdorff_distance_from_one_surface_to_another(point_list_s, point_list_g, new_spacing)
    dis2 = hausdorff_distance_from_one_surface_to_another(point_list_g, point_list_s, new_spacing)
    return max(dis1, dis2)


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = s_volume.sum()
    g_v = g_volume.sum()
    assert(g_v > 0)
    rve = abs(s_v - g_v)/g_v
    return rve

def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if(metric.lower() == "dice"):
        score = binary_dice(s_volume, g_volume)
    elif(metric.lower() == "iou"):
        score = binary_iou(s_volume,g_volume)
    elif(metric.lower() == "hausdorff2d"):
        score = binary_hausdorff2d(s_volume, g_volume, spacing)
    elif(metric.lower() == "hausdorff3d"):
        score = binary_hausdorff3d(s_volume, g_volume, spacing)
    elif(metric.lower() == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))
    return score

def evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    metric = config['metric']
    labels = config['label_list']
    label_convert_source = config.get('label_convert_source', None)
    label_convert_target = config.get('label_convert_target', None)
    s_folder = config['segmentation_folder']
    g_folder = config['ground_truth_folder']
    s_postfix = config.get('segmentation_postfix',None)
    g_postfix = config.get('ground_truth_postfix',None)
    s_format  = config.get('segmentation_format', "nii.gz")
    g_format  = config.get('ground_truth_format', "nii.gz")
    remove_outlier = config.get('remove_outlier', False)

    s_postfix_long = '.' + s_format
    if(s_postfix is not None):
        s_postfix_long = '_' + s_postfix + s_postfix_long
    g_postfix_long = '.' + g_format
    if(g_postfix is not None):
        g_postfix_long = '_' + g_postfix + g_postfix_long

    patient_names_file = config['patient_file_names']
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 
    score_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + s_postfix_long)
        g_name = os.path.join(g_folder, patient_names[i] + g_postfix_long)
        s_dict = load_image_as_4d_array(s_name)
        g_dict = load_image_as_4d_array(g_name)
        s_volume = s_dict["data_array"]; s_spacing = s_dict["spacing"]
        g_volume = g_dict["data_array"]; g_spacing = g_dict["spacing"]
        if((label_convert_source is not None) and label_convert_target is not None):
            s_volume = convert_label(s_volume, label_convert_source, label_convert_target)

        # fuse multiple labels
        s_volume_sub = np.zeros_like(s_volume)
        g_volume_sub = np.zeros_like(g_volume)
        for lab in labels:
            s_volume_sub = s_volume_sub + np.asarray(s_volume == lab, np.uint8)
            g_volume_sub = g_volume_sub + np.asarray(g_volume == lab, np.uint8)

        if(remove_outlier):
            strt = ndimage.generate_binary_structure(3,2) # iterate structure
            post = ndimage.morphology.binary_closing(s_volume_sub, strt)
            post = get_largest_component(post)
            s_volume_sub = np.asarray(post*s_volume_sub, np.uint8)
        
        temp_score = get_evaluation_score(s_volume_sub > 0, g_volume_sub > 0,
                     g_spacing, metric)
        score_all_data.append(temp_score)
        print(patient_names[i], temp_score)
    score_all_data = np.asarray(score_all_data)
    score_mean = [score_all_data.mean(axis = 0)]
    score_std  = [score_all_data.std(axis = 0)]
    np.savetxt("{0:}/{1:}_all_temp.txt".format(s_folder, metric), score_all_data)
    np.savetxt("{0:}/{1:}_mean_temp.txt".format(s_folder, metric), score_mean)
    np.savetxt("{0:}/{1:}_std_temp.txt".format(s_folder, metric), score_std)
    print("{0:} mean ".format(metric), score_mean)
    print("{0:} std  ".format(metric), score_std) 
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/evaluation.py config.cfg')
        exit()
    print("evaluation")
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    evaluation(config_file)
