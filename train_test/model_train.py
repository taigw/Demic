"""Script for training
Author: Guotai Wang
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction as SegmentationLoss
from niftynet.layer.loss_regression import LossFunction as RegressionLoss
from Demic.util.parse_config import parse_config
from Demic.image_io.data_generator import ImageDataGenerator
from Demic.net.net_factory import NetFactory


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

def soft_size_loss(prediction, soft_ground_truth, num_class, weight_map = None):
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])

    pred_size   = tf.reduce_sum(pred, 0)
    ground_size = tf.reduce_sum(ground, 0)
    size_loss   = pred_size - ground_size
    size_loss   = tf.div(size_loss, ground_size + 1e-10)
    size_loss   = tf.square(size_loss)
    size_loss   = tf.reduce_sum(size_loss)
    return size_loss

class TrainAgent(object):
    def __init__(self, config):
        self.config_data = config['tfrecords']
        self.config_net  = config['network']
        self.net_params  = config['network_parameter']
        self.config_train= config['training']
        
        seed = self.config_train.get('random_seed', 1)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    def get_output_and_loss(self):
        pass

    def get_input_output_feed_dict(self):
        pass
    
    def construct_network(self):
        batch_size  = self.config_data.get('batch_size', 5)
        self.full_data_shape = [batch_size] + self.config_net['data_shape']
        self.full_out_shape  = [batch_size] + self.config_net['out_shape']

        self.x = tf.placeholder(tf.float32, shape = self.full_data_shape)
        self.m = tf.placeholder(tf.float32, shape = []) # momentum for batch normalization
        self.get_output_and_loss()

    def create_optimization_step_and_data_generator(self):
        lr = self.config_train.get('learning_rate', 1e-3)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
        with tf.control_dependencies(update_ops):
            self.opt_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            self.data_agent = ImageDataGenerator(self.config_data)
            # create an reinitializable iterator given the dataset structure
            iterator = Iterator.from_structure(self.data_agent.data.output_types,
                                               self.data_agent.data.output_shapes)
            self.next_batch = iterator.get_next()
        # Ops for initializing the two different iterators
        self.training_init_op = iterator.make_initializer(self.data_agent.data)
    
    def train(self):
        # start the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        max_epoch   = self.config_train['maximal_epoch']
        loss_file   = self.config_train['model_save_prefix'] + "_loss.txt"
        start_epoch = self.config_train.get('start_epoch', 0)
        loss_list   = []
        if( start_epoch > 0):
            all_vars = tf.global_variables()
            ignore_var_names = self.config_train.get('ignore_var_names', None)
            if(ignore_var_names is None):
                restore_vars = all_vars
            else:
                restore_vars = []
                for var in all_vars:
                    restore_flag = True
                    for ignore_name in ignore_var_names:
                        if(ignore_name in var.name):
                            restore_flag = False
                            break
                    if(restore_flag):
                        restore_vars.append(var)
            restore_saver = tf.train.Saver(restore_vars)
            restore_saver.restore(self.sess, self.config_train['pretrained_model'])
        
        for epoch in range(start_epoch, max_epoch):
            # Initialize iterator with the training dataset
            self.sess.run(self.training_init_op)
            temp_momentum = float(epoch-start_epoch)/float(max_epoch-start_epoch)
            
            for step in range(self.config_train['batch_number']):
                feed_dict = self.get_input_output_feed_dict()
                feed_dict[self.m] = temp_momentum
                self.opt_step.run(session = self.sess, feed_dict=feed_dict)
            batch_loss_list = []
            
            for test_step in range(self.config_train['test_steps']):
                feed_dict = self.get_input_output_feed_dict()
                feed_dict[self.m] = temp_momentum
                loss_v = self.loss.eval(feed_dict)
                batch_loss_list.append(loss_v)
            batch_loss = np.asarray(batch_loss_list, np.float32).mean()
            print("{0:} Epoch {1:}, loss {2:}".format(datetime.now(), epoch+1, batch_loss))
            # save loss and snapshot
            loss_list.append(batch_loss)
            np.savetxt(loss_file, np.asarray(loss_list))
            if((epoch+1)%self.config_train['snapshot_epoch']  == 0):
                saver.save(self.sess, self.config_train['model_save_prefix']+"_{0:}.ckpt".format(epoch+1))

class SegmentationTrainAgent(TrainAgent):
    def __init__(self, config):
        super(SegmentationTrainAgent, self).__init__(config)
        assert(self.config_data['patch_mode'] == 0 or self.config_data['patch_mode'] == 1)
    
    def get_output_and_loss(self):
        self.class_num = self.config_net['class_num']
        multi_scale_loss = self.config_train.get('multi_scale_loss', False)
        size_constraint  = self.config_train.get('size_constraint', False)
        loss_func = SegmentationLoss(n_class=self.class_num)
        if(multi_scale_loss):
            loss_func1 = SegmentationLoss(n_class=self.class_num)
            loss_func2 = SegmentationLoss(n_class=self.class_num)
            loss_func3 = SegmentationLoss(n_class=self.class_num)
        
        full_weight_shape = [x for x in self.full_out_shape]
        full_weight_shape[-1] = 1
        self.w = tf.placeholder(tf.float32, shape = full_weight_shape)
        self.y = tf.placeholder(tf.int32, shape = self.full_out_shape)
        
        w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        net_class = NetFactory.create(self.config_net['net_type'])
        
        net = net_class(num_classes = self.class_num,
                        parameters  = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        self.predicty = net(self.x, is_training = self.config_net['bn_training'], bn_momentum=self.m)
        print('network output shape ', self.predicty.shape)
        self.loss = loss_func(self.predicty, self.y, weight_map = self.w)
        if(multi_scale_loss):
            y_soft  = get_soft_label(self.y, self.class_num)
            y_pool1 = tf.nn.pool(y_soft, [1, 3, 3], 'AVG', 'VALID', strides = [1, 3, 3])
            predy_pool1 = tf.nn.pool(self.predicty, [1, 3, 3], 'AVG', 'VALID', strides = [1, 3, 3])
            loss1 = soft_dice_loss(predy_pool1, y_pool1, self.class_num)

            y_pool2 = tf.nn.pool(y_soft, [1, 6, 6], 'AVG', 'VALID', strides = [1, 6, 6])
            predy_pool2 = tf.nn.pool(self.predicty, [1, 6, 6], 'AVG', 'VALID', strides = [1, 6, 6])
            loss2 = soft_dice_loss(predy_pool2, y_pool2, self.class_num)

            y_pool3 = tf.nn.pool(y_soft, [1, 12, 12], 'AVG', 'VALID', strides = [1, 12, 12])
            predy_pool3 = tf.nn.pool(self.predicty, [1, 12, 12], 'AVG', 'VALID', strides = [1, 12, 12])
            loss3 = soft_dice_loss(predy_pool3, y_pool3, self.class_num)
            self.loss = (self.loss + loss1 + loss2 + loss3)/4.0
        if(size_constraint):
            print('use size constraint loss')
            y_soft  = get_soft_label(self.y, self.class_num)
            self.loss = self.loss + soft_size_loss(self.predicty, y_soft, self.class_num, weight_map = self.w)
            
    def get_input_output_feed_dict(self):
        [x_batch, w_batch, y_batch] = self.sess.run(self.next_batch)
        feed_dict = {self.x:x_batch, self.w: w_batch, self.y:y_batch}
        return feed_dict

class RegressionTrainAgent(TrainAgent):
    def __init__(self, config):
        super(RegressionTrainAgent, self).__init__(config)
        assert(self.config_data['patch_mode'] == 2)
    
    def get_output_and_loss(self):
        loss_func = RegressionLoss()
        self.y = tf.placeholder(tf.float32, shape = self.full_out_shape)
        
        w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        net_class = NetFactory.create(self.config_net['net_type'])

        output_dim_num = np.prod(self.config_net['out_shape'])
        net = net_class(num_classes = output_dim_num,
                        parameters = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        predicty = net(self.x, is_training = self.config_net['bn_training'], bn_momentum=self.m)
        self.predicty = tf.reshape(predicty, self.full_out_shape)
        print('network output shape ', self.predicty.shape)
        self.loss = loss_func(self.predicty, self.y)
    
    def get_input_output_feed_dict(self):
        [x_batch, y_batch] = self.sess.run(self.next_batch)
        feed_dict = {self.x:x_batch, self.y:y_batch}
        return feed_dict

def model_train(config_file):
    config = parse_config(config_file)
    app_type = config['training']['app_type']
    if(app_type==0):
        train_agent = SegmentationTrainAgent(config)
    else:
        train_agent = RegressionTrainAgent(config)
    train_agent.construct_network()
    train_agent.create_optimization_step_and_data_generator()
    train_agent.train()

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train_test/model_train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
