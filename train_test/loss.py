import tensorflow as tf

def get_loss_function(loss_name):
    if(loss_name == 'cross_entropy'):
        return soft_cross_entropy_loss
    elif(loss_name == "dice" or loss_name == "dice_loss"):
        return soft_dice_loss
    elif(loss_name == "generalized_dice_loss"):
        return generalized_dice_loss
    elif(loss_name == "multi_scale_dice_loss"):
        return multi_scale_soft_dice_loss
    elif(loss_name == "focal_dice_loss"):
        return focal_dice_loss
    elif(loss_name == "focal_gdl_loss"):
        return focal_gdl_loss
    else:
        raise ValueError("unsupported loss {0:}".format(loss_name))

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [N, D, H, W, 1]
        output_tensor: shape [N, D, H, W, num_class]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = tf.equal(input_tensor, i*tf.ones_like(input_tensor,tf.float32))
        tensor_list.append(temp_prob)
    output_tensor = tf.concat(tensor_list, axis=-1)
    output_tensor = tf.cast(output_tensor, tf.float32)
    return output_tensor

def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None, softmax = True):
    pred   = tf.reshape(prediction, [-1, num_class])
    if(softmax):
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
    dice_score = (2.0*intersect + 1.0)/(ref_vol + seg_vol + 1.0)
    dice_score = tf.reduce_mean(dice_score)
    return 1.0-dice_score*0.99

def generalized_dice_loss(prediction, soft_ground_truth, num_class, 
       weight_map=None, softmax = True, class_weight = [1.0, 1.0]):
    assert(len(class_weight) == num_class)
    pred   = tf.reshape(prediction, [-1, num_class])
    if(softmax):
    	pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    n_voxels = ground.get_shape()[0].value
    cls_w = tf.constant(class_weight)
    if(weight_map is not None):
        weight_map = tf.reshape(weight_map, [-1])
        weight_map_nclass = tf.reshape(
            tf.tile(weight_map, [num_class]), pred.get_shape())
        ref_vol   = tf.reduce_sum(weight_map_nclass*ground, 0)*cls_w
        intersect = tf.reduce_sum(weight_map_nclass*ground*pred, 0)*cls_w
        seg_vol   = tf.reduce_sum(weight_map_nclass*pred, 0)*cls_w
    else:
        ref_vol   = tf.reduce_sum(ground, 0)*cls_w
        intersect = tf.reduce_sum(ground*pred, 0)*cls_w
        seg_vol   = tf.reduce_sum(pred, 0)*cls_w
    numerator   = tf.reduce_sum(intersect) + 1.0
    denominator = tf.reduce_sum(ref_vol + seg_vol) + 1.0
    gdl_loss    = 1.0 - 2*numerator/denominator
    return gdl_loss

def focal_gdl_loss(prediction,  soft_ground_truth, num_class, alpha = 1.0, 
       weight_map=None, softmax = True, class_weight = [1.0, 1.0]):
    assert(len(class_weight) == num_class)
    [N, D, H, W, C] = prediction.get_shape().as_list()
    if(softmax):
        prediction = tf.nn.softmax(prediction)
    intersect = 1.0
    union     = 1.0
    for n in range(N):
        for c in range(C):
            pred   = tf.slice(prediction, begin = [n, 0, 0, 0, c], 
                                              size =[1, D, H, W, 1])
            ground = tf.slice(soft_ground_truth, begin = [n, 0, 0, 0, c], 
                                              size =[1, D, H, W, 1])
            wght = tf.math.abs(pred - ground)
            wght = tf.ones_like(pred)*alpha + (1.0 - alpha) * wght
            
            intersect += tf.reduce_sum(pred * ground * wght)
            union     += tf.reduce_sum((pred + ground) * wght)
    loss = 1.0 - 2*intersect/union
    return loss

def focal_dice_loss(prediction, soft_ground_truth, num_class, alpha = 1.0, weight_map=None, softmax = True):
    [N, D, H, W, C] = prediction.get_shape().as_list()
    if(softmax):
        prediction = tf.nn.softmax(prediction)
    loss_sum = 0.0
    weight_sum = 0.0
    for n in range(N):
        for c in range(C):
            pred   = tf.slice(prediction, begin = [n, 0, 0, 0, c], 
                                              size =[1, D, H, W, 1])
            ground = tf.slice(soft_ground_truth, begin = [n, 0, 0, 0, c], 
                                              size =[1, D, H, W, 1])
            wght = tf.math.abs(pred - ground)
            wght = tf.ones_like(pred)*alpha + (1.0 - alpha) * wght

            loss   =  soft_dice_loss(pred, ground, 1, 
                      weight_map = wght,softmax = False)
            loss_sum   += loss 
    return loss_sum/(N*C)

def multi_scale_soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map = None):
    loss = soft_dice_loss(prediction, soft_ground_truth, num_class)
    y_pool1 = tf.nn.pool(soft_ground_truth, [1, 2, 2], 'AVG', 'VALID', strides = [1, 2, 2])
    predy_pool1 = tf.nn.pool(prediction, [1, 2, 2], 'AVG', 'VALID', strides = [1, 2, 2])
    loss1 =  soft_dice_loss(predy_pool1, y_pool1, num_class)

    y_pool2 = tf.nn.pool(soft_ground_truth, [1, 4, 4], 'AVG', 'VALID', strides = [1, 4, 4])
    predy_pool2 = tf.nn.pool(prediction, [1, 4, 4], 'AVG', 'VALID', strides = [1, 4, 4])
    loss2 =  soft_dice_loss(predy_pool2, y_pool2, num_class)
    loss = (loss + loss1 + loss2 )/3.0

def soft_cross_entropy_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    ce = ground* tf.log(pred)
    if(weight_map is not None):
        n_voxels = tf.reduce_sum(weight_map)
        weight_map = tf.reshape(weight_map, [-1])
        weight_map_nclass = tf.reshape(
            tf.tile(weight_map, [num_class]), pred.get_shape())
        ce = ce * weight_map_nclass
    ce = -tf.reduce_mean(ce)
    return ce


def soft_size_loss(prediction, soft_ground_truth, num_class, weight_map = None):
    pred = tf.reshape(prediction, [-1, num_class])
    pred = tf.nn.softmax(pred)
    grnd = tf.reshape(soft_ground_truth, [-1, num_class])

    pred_size = tf.reduce_mean(pred, 0)
    grnd_size = tf.reduce_mean(grnd, 0)
    size_loss = tf.square(pred_size - grnd_size)
    size_loss = tf.multiply(size_loss, tf.constant([0] + [1]*(num_class-1), tf.float32))
    size_loss = tf.reduce_sum(size_loss)/(num_class - 1)
    return size_loss

