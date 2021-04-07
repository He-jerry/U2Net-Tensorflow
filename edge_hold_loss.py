import tensorflow as tf
from keras import backend as K
from keras.backend.common import epsilon
from segmentation_models.losses import *

dice_loss = DiceLoss()

binary_focal_loss = BinaryFocalLoss()


binary_crossentropy = BinaryCELoss()

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def logit(inputs):
    _epsilon = _to_tensor(epsilon(), inputs.dtype.base_dtype)
    inputs = tf.clip_by_value(inputs, _epsilon, 1 - _epsilon)
    inputs = tf.log(inputs / (1 - inputs))
    return inputs

def tfLaplace(x):
    laplace = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32)
    laplace = tf.reshape(laplace, [3, 3, 1, 1])
    edge = tf.nn.conv2d(x, laplace, strides=[1, 1, 1, 1], padding='SAME')
    edge = tf.nn.relu(tf.tanh(edge))
    return edge

def EdgeLoss(y_true, y_pred):
    y_true_edge = tfLaplace(y_true)
    edge_pos = 2.
    edge_loss = K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true_edge,y_pred,edge_pos), axis=-1)
    return edge_loss

def EdgeHoldLoss(y_true, y_pred):
    y_pred2 = tf.sigmoid(y_pred)
    y_true_edge = tfLaplace(y_true)
    y_pred_edge = tfLaplace(y_pred2)
    y_pred_edge = logit(y_pred_edge)
    edge_loss = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_edge,logits=y_pred_edge), axis=-1)
    saliency_pos = 1.12
    saliency_loss = K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,saliency_pos), axis=-1)
    return 0.7*saliency_loss+0.3*edge_loss

"""
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()
"""

def totalloss(y_true, y_pred):
    y_pred2=tf.sigmoid(y_pred)
    ssim_loss=1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred2, max_val=1.0))
    bce_loss0 = BinaryCELoss()
    jaccard_loss = JaccardLoss()
    loss0 = bce_loss0(y_true,y_pred2)
    return EdgeHoldLoss(y_true,y_pred)+dice_loss(y_true,y_pred2)+ssim_loss+loss0+jaccard_loss(y_true,y_pred2)

def structure_loss(y_true,y_pred):
    y_pred2 = tf.sigmoid(y_pred)
    poolvalue=tf.keras.layers.AvgPool2D((31,31),strides=1,padding='same')(y_true)
    weit=1+5*K.abs(poolvalue-y_true)
    wbce=tf.nn.sigmoid_cross_entropy_with_logits(y_pred,y_true)
    bceloss=tf.reduce_sum(wbce*weit)/tf.reduce_sum(weit)

    inter=tf.reduce_sum((y_pred2*y_true)*weit)
    union=tf.reduce_sum((y_pred2+y_true)*weit)
    wiou=1-(inter+1)/(union-inter+1)

    return tf.reduce_mean(wbce+wiou)







