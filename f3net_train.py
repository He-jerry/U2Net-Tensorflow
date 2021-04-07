from keras import callbacks, optimizers
import tensorflow as tf
import os
from keras.layers import Input
# from keras.optimizers import SGD
from keras.optimizers import SGD, Adam
from model import VGG16
from reimp import f3net
from data import getTrainGenerator
from utils import *
from edge_hold_loss import *
from segmentation_models import metrics as me
import math

binary_focal_dice_loss = binary_focal_loss + dice_loss

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_scheduler(epoch):
    drop = 0.5
    epoch_drop = epochs / 8.
    lr = base_lr * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
    print('lr: %f' % lr)
    return lr


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train model your dataset')
    parser.add_argument('--model_weights', default="model/f3net_00050.h5", help='your pretrained model weights', type=str)
    parser.add_argument('--lr', default="0.001", help='your model weights', type=str)
    parser.add_argument('--epoch', default="50", help='your model weights', type=str)

    args = parser.parse_args()

    model_name = args.model_weights


    print("model_weights", model_name)

    target_size = (256, 256)
    batch_size = 4
    base_lr = 0.01
    epochs = 50

    train_path='sod/imgs/'
    mask_path='sod/masks/'

    steps_per_epoch = 1000 / batch_size

    # optimizer = optimizers.SGD(lr=base_lr, momentum=0.9, decay=0)
    optimizer = optimizers.Adam(lr=base_lr)
    # optimizer=tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    loss = EdgeHoldLoss

    metrics = [acc, pre, rec, me.iou_score, me.f1_score]
    dropout = True
    with_CPFE = True
    with_CA = True
    with_SA = True
    log = './PFA.csv'
    tb_log = './tensorboard-logs/PFA'
    model_save = 'F3Net_keras/'
    os.makedirs(model_save)
    model_save_period = 10

    if target_size[0] % 32 != 0 or target_size[1] % 32 != 0:
        raise ValueError('Image height and wight must be a multiple of 32')

    traingen = getTrainGenerator(train_path,mask_path, target_size, batch_size, israndom=True)

    model_input = Input(shape=(target_size[0], target_size[1], 3))
    model = f3net(model_input)

    tb = callbacks.TensorBoard(log_dir=tb_log)
    lr_decay = callbacks.LearningRateScheduler(schedule=lr_scheduler)
    es = callbacks.EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto')
    modelcheck = callbacks.ModelCheckpoint(model_save + '{epoch:05d}.h5', monitor='loss', verbose=1,
                                           save_best_only=False, save_weights_only=True, mode='auto',
                                           period=model_save_period)
    callbacks = [lr_decay, modelcheck, tb]
    #pred1, pred2, out2h, out3h, out4h, out5h
    model.compile(optimizer=Adam(lr=base_lr),loss={'pred1': totalloss, 'pred2': totalloss, 'out2h': totalloss, 'out3h': totalloss, 'out4h': totalloss,'out5h': totalloss}, metrics=metrics)
    model.fit_generator(traingen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks)