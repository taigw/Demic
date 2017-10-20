"""Script for training
Author: Guotai Wang
"""
import os
import sys
import numpy as np
import tensorflow as tf
from util.parse_config import parse_config
from image_io.data_generator import ImageDataGenerator
from net.net_factory import NetFactory
from tensorflow.contrib.data import Iterator

def model_train(config_file):
    config = parse_config(config_file)
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']

    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    full_data_shape  = config_net['data_shape']
    full_label_shape = config_net['label_shape']
    data_channel= data_shape[-1]
    class_num   = label_shape[-1]
    batch_size  = config_data.get('tfrecords', 5)
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

    num_epochs  = config_train('epoch')
    train_batches_per_epoch = config_train('batch_number')
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            [img_batch, weight_batch, label_batch] = sess.run(next_batch)
            opt_step.run(session = sess, feed_dict={x:img_batch, w: weight_batch, y:label_batch})

if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('Number of arguments should be 2. e.g.')
        print('    python model_train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
