"""Script for training
Author: Guotai Wang
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from Demic.util.parse_config import parse_config
from Demic.image_io.data_generator import ImageDataGenerator
from Demic.image_io.file_read_write import save_array_as_nifty_volume
from Demic.net.net_factory import NetFactory
from Demic.train_test.loss import get_soft_label, get_loss_function

class TrainAgent(object):
    def __init__(self, config):
        self.config_data    = config['dataset']
        self.config_sampler = config['sampler']
        self.config_net     = config['network']
        self.net_params     = config['network_parameter']
        self.config_train   = config['training']
        
        seed = self.config_train.get('random_seed', 1)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def construct_network(self):
        batch_size  = self.config_sampler.get('batch_size', 5)
        self.full_data_shape = [batch_size] + self.config_sampler['data_shape']
        self.full_out_shape  = [batch_size] + self.config_sampler['label_shape']
        self.class_num       = self.config_net['class_num']
        net_class            = NetFactory.create(self.config_net['net_type'])
        loss_func = get_loss_function(self.config_train.get('loss_function', 'dice'))
        
        full_weight_shape = [x for x in self.full_out_shape]
        full_weight_shape[-1] = 1
        self.x = tf.placeholder(tf.float32, shape = self.full_data_shape)
        self.w = tf.placeholder(tf.float32, shape = full_weight_shape)
        self.y = tf.placeholder(tf.int32, shape = self.full_out_shape)
        
        w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        
        net = net_class(num_classes = self.class_num,
                        parameters  = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        predicty = net(self.x, is_training = self.config_net['bn_training'])
        self.predicty = tf.reshape(predicty, self.full_out_shape[:-1] + [self.class_num])
        print('network output shape ', self.predicty.shape)
        y_soft  = get_soft_label(self.y, self.class_num)
        loss = loss_func(self.predicty, y_soft, self.class_num, weight_map = self.w)
        self.loss = loss
        
        pred   = tf.cast(tf.argmax(self.predicty, axis = -1), tf.int32)
        y_reshape = tf.reshape(self.y, tf.shape(pred))
        intersect = tf.cast(tf.reduce_sum(pred * y_reshape), tf.float32)
        volume_sum = tf.cast(tf.reduce_sum(pred) + tf.reduce_sum(y_reshape), tf.float32)
        self.dice = 2.0*intersect/(volume_sum + 1.0)

        # add tf scalar
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('dice', self.dice)

    def get_variable_list(self, var_names, include = True):
        all_vars = tf.global_variables()
        output_vars = []
        for var in all_vars:
            if(include == False):
                output_flag = True
                if(var_names is not None):
                    for ignore_name in var_names:
                        if(ignore_name in var.name):
                            output_flag = False
                            break
            else:
                output_flag = False
                for include_name in var_names:
                    if(include_name in var.name):
                        output_flag = True
                        break
            if(output_flag):
                output_vars.append(var)
        return output_vars
    
    def get_input_output_feed_dict(self, stage, net_idx = 0):
        while(True):
            if(stage == 'train'):
                try:
                    [x_batch, w_batch, y_batch] = self.sess.run(self.next_train_batch)
                    if (x_batch.shape[0] == self.config_sampler.get('batch_size', 5)):
                        break
                    else:
                        self.sess.run(self.train_init_op)
                except tf.errors.OutOfRangeError:
                    self.sess.run(self.train_init_op)
            else:
                try:
                    [x_batch, w_batch, y_batch] = self.sess.run(self.next_valid_batch[net_idx])
                    if (x_batch.shape[0] == self.config_sampler.get('batch_size', 5)):
                        break
                    else:
                        self.sess.run(self.valid_init_op[net_idx])
                except tf.errors.OutOfRangeError:
                    self.sess.run(self.valid_init_op[net_idx])
        feed_dict = {self.x:x_batch, self.w: w_batch, self.y:y_batch}
        return feed_dict

    def create_optimization_step_and_data_generator(self):
        learn_rate  = self.config_train.get('learning_rate', 1e-3)
        vars_fixed  = self.config_train.get('vars_not_update', None)
        vars_update = self.get_variable_list(vars_fixed, include = False)
        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
        with tf.control_dependencies(update_ops):
            self.opt_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, var_list = vars_update)
        
        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            self.train_data = ImageDataGenerator(self.config_data['data_train'], self.config_sampler)
            # create an reinitializable iterator given the dataset structure
            train_iterator = Iterator.from_structure(self.train_data.data.output_types,
                                               self.train_data.data.output_shapes)
            self.next_train_batch = train_iterator.get_next()
        # Ops for initializing the two different iterators
        self.train_init_op = train_iterator.make_initializer(self.train_data.data)
        valid_data       = []
        next_valid_batch = []
        valid_init_op    = []
        for i in range(len(self.config_data) - 1):
            with tf.device('/cpu:0'):
                temp_valid_data = ImageDataGenerator(self.config_data["data_valid{0:}".format(i)], self.config_sampler)
                temp_valid_iterator = Iterator.from_structure(temp_valid_data.data.output_types,
                                            temp_valid_data.data.output_shapes)
                temp_next_valid_batch = temp_valid_iterator.get_next()
            temp_valid_init_op = temp_valid_iterator.make_initializer(temp_valid_data.data)
            valid_data.append(temp_valid_data)
            next_valid_batch.append(temp_next_valid_batch)
            valid_init_op.append(temp_valid_init_op)
        self.valid_data = valid_data
        self.next_valid_batch = next_valid_batch
        self.valid_init_op = valid_init_op

    def save_batch_data(self, feed_dict, iter):   
            x_batch = feed_dict[self.x]
            save_array_as_nifty_volume(x_batch[0], '../temp/img{0:}.nii.gz'.format(iter))
            y_batch = feed_dict[self.y]
            save_array_as_nifty_volume(y_batch[0], '../temp/img{0:}_lab.nii.gz'.format(iter))

    def train(self):
        # start the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        save_vars = self.get_variable_list([self.config_net['net_name']], include = True)
        saver = tf.train.Saver(save_vars)
        
        start_iter  = self.config_train.get('start_iter', 0)
        max_iter    = self.config_train['maximal_iter']
        loss_file   = self.config_train['model_save_prefix'] + "_loss.txt"
        dice_file   = self.config_train['model_save_prefix'] + "_dice.txt"
        
        loss_list, dice_list   = [], []
        if(start_iter > 0):
            restore_vars  = save_vars 
            restore_saver = tf.train.Saver(restore_vars)
            restore_saver.restore(self.sess, self.config_train['pretrained_model'])
 
            loss_list = list(np.loadtxt(loss_file)[:start_iter - 1])
            dice_list = list(np.loadtxt(loss_file)[:start_iter - 1])
        
        # make sure the graph is fixed during training
        summ_merged = tf.summary.merge_all() # for tensorboard 
        tf.get_default_graph().finalize()
        train_summ_writer = tf.summary.FileWriter(
            self.config_train['model_save_prefix'] + "/train", self.sess.graph)
        valid_summ_writers = []
        for i in range(len(self.valid_data)):
            valid_summ_writer = tf.summary.FileWriter(
                self.config_train['model_save_prefix'] + "/valid{0:}".format(i), self.sess.graph)
            valid_summ_writers.append(valid_summ_writer)

        self.sess.run(self.train_init_op)
        for valid_init_op in self.valid_init_op:
            self.sess.run(valid_init_op)
        for iter in range(start_iter, max_iter):
            try:
                feed_dict = self.get_input_output_feed_dict('train')
                self.opt_step.run(session = self.sess, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                self.sess.run(self.training_init_op)
            # save_batch_data(feed_dict, iter)

            if(iter==start_iter or ((iter + 1) % self.config_train['test_interval'] == 0)):
                feed_dict = self.get_input_output_feed_dict('train')
                [loss_v, dice_v, merged_value] = self.sess.run(
                    [self.loss, self.dice, summ_merged],feed_dict = feed_dict)
                batch_loss = [iter + 1, loss_v]
                batch_dice = [iter + 1, dice_v]
                train_summ_writer.add_summary(merged_value, iter)

                for valid_idx in range(len(self.config_data) - 1):
                    feed_dict = self.get_input_output_feed_dict('valid', valid_idx)
                    [loss_v, dice_v, merged_value] = self.sess.run(
                        [self.loss, self.dice, summ_merged],feed_dict = feed_dict)
                    batch_loss.append(loss_v)
                    batch_dice.append(dice_v)
                    valid_summ_writers[valid_idx].add_summary(merged_value, iter)
                if(not((iter == start_iter) and (iter > 0))):
                    loss_list.append(batch_loss)
                    dice_list.append(batch_dice)

                if(iter == start_iter or (iter + 1) %  self.config_train['display_interval'] == 0):
                    loss_str = ','.join(["{0:.4f}".format(item) for item in batch_loss[1:]])
                    dice_str = ','.join(["{0:.4f}".format(item) for item in batch_dice[1:]])
                    print("{0:} Iter {1:}, loss {2:}, dice {3:}".format(datetime.now(), \
                          iter+1, loss_str, dice_str))
                    np.savetxt(loss_file, np.asarray(loss_list))
                    np.savetxt(dice_file, np.asarray(dice_list))
            if((iter+1)%self.config_train['snapshot_iter']  == 0):
                saver.save(self.sess, self.config_train['model_save_prefix']+"_{0:}.ckpt".format(iter+1))
       

def model_train(config_file):
    config = parse_config(config_file)
    app_type = config['training']['app_type']
    if(app_type==0):
        train_agent = TrainAgent(config)
    else:
        raise ValueError("unsupported application type")
    train_agent.construct_network()
    train_agent.create_optimization_step_and_data_generator()
    train_agent.train()

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train_test/model_train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
