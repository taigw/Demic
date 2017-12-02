"""Script for training
Author: Guotai Wang
"""
import os
import sys
import random
import nibabel
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.data import Iterator
from niftynet.layer.loss_segmentation import LossFunction as SegmentationLoss
from Demic.image_io.image_loader import ImageLoader
from Demic.net.net_factory import NetFactory

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [N, D, H, W, 1]
        output_tensor: shape [N, D, H, W, num_class]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = tf.equal(input_tensor, i*tf.ones_like(input_tensor,tf.int32))
        tensor_list.append(temp_prob)
    output_tensor = tf.concat(tensor_list, axis=-1)
    output_tensor = tf.cast(output_tensor, tf.float32)
    return output_tensor

def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    n_voxels = ground.get_shape()[0].value
    if(weight_map is not None):
        weight_map = tf.reshape(weight_map, [-1])
        weight_map_nclass = tf.reshape(
            tf.tile(weight_map, [num_class]), pred.get_shape())
        ref_vol = tf.reduce_sum(weight_map_nclass*ground, 0)
        intersect = tf.reduce_sum(weight_map_nclass*ground*pred, 0)
        seg_vol = tf.reduce_sum(weight_map_nclass*pred, 0)
    else:
        ref_vol = tf.reduce_sum(ground, 0)
        intersect = tf.reduce_sum(ground*pred, 0)
        seg_vol = tf.reduce_sum(pred, 0)
    dice_numerator = 2*tf.reduce_sum(intersect)
    dice_denominator = tf.reduce_sum(seg_vol + ref_vol)
    dice_score = dice_numerator/dice_denominator
    return 1-dice_score

class TrainAgent(object):
    def __init__(self, config):
        self.config_data = config['data']
        self.config_net  = config['network']
        self.net_params  = config['network_parameter']
        self.config_train= config['training']
        
        seed = self.config_train.get('random_seed', 1)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def get_input_output_feed_dict(self, is_training):
        if(is_training):
            one_batch = self.sess.run(self.next_train_batch)
        else:
            one_batch = self.sess.run(self.next_valid_batch)
        feed_dict = {self.x:one_batch['image'],
                     self.w:one_batch['weight'],
                     self.y:one_batch['label']}
        return feed_dict

    def construct_network(self):
        # 1, get input and output shape
        self.batch_size      = self.config_train.get('batch_size', 5)
        self.full_data_shape = [self.batch_size] + self.config_net['patch_shape_x']
        self.full_out_shape  = [self.batch_size] + self.config_net['patch_shape_y']

        self.x = tf.placeholder(tf.float32, shape = self.full_data_shape)
        self.w = tf.placeholder(tf.float32, shape = self.full_out_shape)
        self.y = tf.placeholder(tf.int32, shape = self.full_out_shape)
        self.m = tf.placeholder(tf.float32, shape = []) # momentum for batch normalization
        
        # 2, define network
        w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        if(type(self.config_net['net_type']) is str):
            net_class = NetFactory.create(self.config_net['net_type'])
        else:
            print('customized network is used')
            net_class = self.config_net['net_type']
        self.class_num = self.config_net['class_num']
        net = net_class(num_classes = self.class_num,
                        parameters  = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        self.predicty = net(self.x, is_training = self.config_net['bn_training'], bn_momentum=self.m)
        print('network output shape ', self.predicty.shape)
        self.set_loss_and_optimizer()
    
    def set_loss_and_optimizer(self):
        # 1, set loss function
        loss_type = self.config_train.get('loss_type', 'Dice')
        loss_func = SegmentationLoss(self.class_num, loss_type)
        self.loss = loss_func(self.predicty, self.y, weight_map = self.w)
        
        # 2, set optimizer
        lr = self.config_train.get('learning_rate', 1e-3)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
        with tf.control_dependencies(update_ops):
            self.opt_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
            
    def create_data_generator(self):
        def get_data_type(typename):
            return tf.float32 if(typename == 'float') else tf.int32
        x_type = get_data_type(self.config_net['patch_type_x'])
        y_type = get_data_type(self.config_net['patch_type_y'])
        self.config_data['data_type'] = {
                'features': {'x': x_type, 'w': x_type},
                'labels':   {'y': y_type}}
        self.config_data['data_shape'] = {
                'features': {'x': self.config_net['patch_shape_x'],
                             'w': self.config_net['patch_shape_y']},
                'labels':   {'y': self.config_net['patch_shape_y']}}
        self.config_data['batch_size'] = self.batch_size
        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            self.data_agent = ImageLoader(self.config_data)
            # create an reinitializable iterator given the dataset structure
            train_dataset = self.data_agent.get_dataset('train')
            valid_dataset = self.data_agent.get_dataset('valid')
            train_iterator = Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
            valid_iterator = Iterator.from_structure(valid_dataset.output_types,
                                               valid_dataset.output_shapes)
            self.next_train_batch = train_iterator.get_next()
            self.next_valid_batch = valid_iterator.get_next()
        # Ops for initializing the two different iterators
        self.train_init_op = train_iterator.make_initializer(train_dataset)
        self.valid_init_op = valid_iterator.make_initializer(valid_dataset)
    
    def train(self):
        # 1, construct network and create data generator
        self.construct_network()
        self.create_data_generator()
        
        # 2, start the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        max_epoch   = self.config_train['maximal_epoch']
        loss_file   = self.config_train['model_save_prefix'] + "_loss.txt"
        start_epoch = self.config_train.get('start_epoch', 0)
        loss_list   = []
        if( start_epoch> 0):
            saver.restore(self.sess, self.config_train['pretrained_model'])
        
        for epoch in range(start_epoch, max_epoch):
            # 3, Initialize iterators and train for one epoch
            temp_momentum = float(epoch)/float(max_epoch)
            train_loss_list = []
            self.sess.run(self.train_init_op)
            for step in range(self.config_train['batch_number']):
                feed_dict = self.get_input_output_feed_dict(is_training = True)
                feed_dict[self.m] = temp_momentum
                self.opt_step.run(session = self.sess, feed_dict=feed_dict)
                if(step < self.config_train['test_steps']):
                    loss_train = self.loss.eval(feed_dict)
                    train_loss_list.append(loss_train)
            batch_loss = np.asarray(train_loss_list, np.float32).mean()
            epoch_loss = [batch_loss]
            
            if(self.config_data.get('data_names_val', None) is not None):
                valid_loss_list = []
                self.sess.run(self.valid_init_op)
                for test_step in range(self.config_train['test_steps']):
                    feed_dict = self.get_input_output_feed_dict(is_training = False)
                    feed_dict[self.m] = temp_momentum
                    loss_valid = self.loss.eval(feed_dict)
                    valid_loss_list.append(loss_valid)
                batch_loss = np.asarray(valid_loss_list, np.float32).mean()
                epoch_loss.append(batch_loss)
                
            print("{0:} Epoch {1:}, loss {2:}".format(datetime.now(), epoch+1, epoch_loss))
            
            # 4, save loss and snapshot
            loss_list.append(epoch_loss)
            np.savetxt(loss_file, np.asarray(loss_list))
            if((epoch+1)%self.config_train['snapshot_epoch']  == 0):
                saver.save(self.sess, self.config_train['model_save_prefix']+"_{0:}.ckpt".format(epoch+1))

