from keras.models import *
from bilinear_upsampling import BilinearUpsampling
import tensorflow as tf
from keras import backend as K

from attention import *
import keras
class BatchNorm(BatchNormalization):
    def call(self, inputs, training=None):
          return super(self.__class__, self).call(inputs, training=True)
def BN(input_tensor):
    bn = BatchNorm()(input_tensor)
    a = Activation('relu')(bn)
    return a


#keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest", **kwargs)
def basicblocks(input,filter,dilates=1):
    x1=Conv2D(filter, (3, 3), padding='same',dilation_rate=1*dilates)(input)
    x1=BN(x1)
    #x1=Activation('relu',name='block1_relu')
    return x1
def RSU7(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    #6
    hx6=basicblocks(hx,mid_ch,1)
    #7
    hx7=basicblocks(hx6,mid_ch,2)

    #down
    #6
    hx6d=Concatenate(axis=-1)([hx7,hx6])
    hx6d=basicblocks(hx6d,mid_ch,1)
    a,b,c,d=K.int_shape(hx5)
    hx6d=keras.layers.UpSampling2D(size=(2,2))(hx6d)

    #5
    hx5d = Concatenate(axis=-1)([hx6d, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    #model=Model(inputs=input, outputs=output, name="rsu7")
    return output

def RSU6(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    #6
    hx6=basicblocks(hx,mid_ch,1)
    hx6=keras.layers.UpSampling2D((2, 2))(hx6)

    #5
    hx5d = Concatenate(axis=-1)([hx6, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    #model=Model(inputs=input, outputs=output, name="rsu6")
    return output


def RSU5(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    #hx5 = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    hx5 = keras.layers.UpSampling2D((2, 2))(hx5)
    # 4
    hx4d = Concatenate(axis=-1)([hx5, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    #model=Model(inputs=input, outputs=output, name="rsu5")
    return output


def RSU4(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx4=keras.layers.UpSampling2D((2,2))(hx4)

    # 3
    hx3d = Concatenate(axis=-1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    #model=Model(inputs=input, outputs=output, name="rsu4")
    return output

def RSU4f(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    #2
    hx2=basicblocks(hx,mid_ch,2)
    #3
    hx3 = basicblocks(hx, mid_ch, 4)
    #4
    hx4=basicblocks(hx,mid_ch,8)

    # 3
    hx3d = Concatenate(axis=-1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 4)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 2)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    #model=Model(inputs=input, outputs=output, name="rsu4f")
    return output


def u2net(input,in_ch=3,out_ch=1):
    stage1=RSU7(input,in_ch=3,mid_ch=32,out_ch=64)
    stage1p=keras.layers.MaxPool2D((2,2),strides=2)(stage1)
    stage2=RSU6(stage1p,in_ch=64,mid_ch=32,out_ch=128)
    stage2p=keras.layers.MaxPool2D((2,2),strides=2)(stage2)
    stage3 = RSU5(stage2p, in_ch=128, mid_ch=64, out_ch=256)
    stage3p = keras.layers.MaxPool2D((2,2),strides=2)(stage3)
    stage4 = RSU4(stage3p, in_ch=256, mid_ch=128, out_ch=512)
    stage4p = keras.layers.MaxPool2D((2,2),strides=2)(stage4)
    stage5 = RSU4f(stage4p, in_ch=512, mid_ch=256, out_ch=512)
    stage6=RSU4f(stage5,in_ch=512,mid_ch=256,out_ch=512)
    stage6u=keras.layers.UpSampling2D((1,1))(stage6)


    #decoder
    stage6a=Concatenate(axis=-1)([stage6u,stage5])
    stage5d=RSU4f(stage6a,1024,256,512)
    stage5du=keras.layers.UpSampling2D((2,2))(stage5d)

    stage5a = Concatenate(axis=-1)([stage5du, stage4])
    stage4d = RSU4(stage5a, 1024, 128,256)
    stage4du = keras.layers.UpSampling2D((2,2))(stage4d)

    stage4a = Concatenate(axis=-1)([stage4du, stage3])
    stage3d = RSU5(stage4a, 512,64,128)
    stage3du = keras.layers.UpSampling2D((2,2))(stage3d)

    stage3a = Concatenate(axis=-1)([stage3du, stage2])
    stage2d = RSU6(stage3a, 256,32,64)
    stage2du = keras.layers.UpSampling2D((2,2))(stage2d)

    stage2a = Concatenate(axis=-1)([stage2du, stage1])
    stage1d = RSU6(stage2a, 128,16,64)

    #side output
    side1=Conv2D(out_ch,(3,3),padding='same',name='side1')(stage1d)
    side2=Conv2D(out_ch,(3,3),padding='same')(stage2d)
    side2=keras.layers.UpSampling2D((2,2),name='side2')(side2)
    side3 = Conv2D(out_ch, (3, 3), padding='same')(stage3d)
    side3 = keras.layers.UpSampling2D((4,4),name='side3')(side3)
    side4 = Conv2D(out_ch, (3, 3), padding='same')(stage4d)
    side4 = keras.layers.UpSampling2D((8,8),name='side4')(side4)
    side5 = Conv2D(out_ch, (3, 3), padding='same')(stage5d)
    side5 = keras.layers.UpSampling2D((16,16),name='side5')(side5)
    side6 = Conv2D(out_ch, (3, 3), padding='same')(stage6)
    side6 = keras.layers.UpSampling2D((16,16),name='side6')(side6)
    out=Concatenate(axis=-1)([side1,side2,side3,side4,side5,side6])
    out=Conv2D(out_ch,(1,1),padding='same',name='out')(out)

    model=Model(input=input,output=[side1,side2,side3,side4,side5,side6,out])
    return model

