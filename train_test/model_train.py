"""Script for training
Author: Guotai Wang
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.parse_config import parse_config
from image_io.data_generator import ImageDataGenerator
from net.net_factory import NetFactory
from tensorflow.contrib.data import Iterator

def model_train(config_file):
    config = parse_config(config_file)
    config = parse_config(config_file)
    config_tfrecords = config['tfrecords']
    config_net       = config['network']
    config_train     = config['training']

    random.seed(config_train.get('random_seed', 1))
    assert(config_tfrecords['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    batch_size  = config_tfrecords.get('batch_size', 5)
    full_data_shape  = [batch_size] + config_net['data_shape']
    full_label_shape = [batch_size] + config_net['label_shape']
    data_channel= full_data_shape[-1]
    class_num   = config_net['class_num']
    
    full_weight_shape = [i for i in full_data_shape]
    full_weight_shape[-1] = 1
    data_space_shape  = full_data_shape[:-1]
    label_space_shape = full_label_shape[:-1]
    
    # construct graph
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_weight_shape)
    y = tf.placeholder(tf.int32, shape = full_label_shape)

    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    predicty = net(x, is_training = True)
    print(predicty)
    # define loss function and optimization method
    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(config_tfrecords)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()
    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)

    # start the session 
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    maximal_epoch  = config_train['maximal_epoch']
    train_batches_per_epoch = config_train['batch_number']
    test_steps = config_train['test_steps']
    loss_list = []
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_epoch = config_train.get('start_epoch', 0)
    if( start_epoch> 0):
        saver.restore(sess, config_train['pretrained_model'])

    for epoch in range(start_epoch, maximal_epoch):
        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        for step in range(train_batches_per_epoch):
            [img_batch, weight_batch, label_batch] = sess.run(next_batch)
            opt_step.run(session = sess, feed_dict={x:img_batch, w: weight_batch, y:label_batch})
        batch_loss_list = []
        for test_step in range(test_steps):
            [img_batch, weight_batch, label_batch] = sess.run(next_batch)
            loss_v = loss.eval(feed_dict ={x:img_batch, w:weight_batch, y:label_batch})
            batch_loss_list.append(loss_v)
        batch_loss = np.asarray(batch_loss_list, np.float32).mean()
        print("{0:} Epoch {1:}, loss {2:}".format(datetime.now(), epoch+1, batch_loss))
        # save loss and snapshot
        loss_list.append(batch_dice)
        np.savetxt(loss_file, np.asarray(loss_list))
        if((epoch+1)%config_train['snapshot_epoch']  == 0):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(epoch+1))

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python model_train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
