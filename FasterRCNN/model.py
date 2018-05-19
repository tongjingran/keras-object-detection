from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.models import Model
from keras.layers import TimeDistributed, Dense, Dropout, Flatten, Conv2D, Lambda, Input, Lambda
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from keras.engine.topology import Layer
from keras.applications.vgg16 import VGG16
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import tensorflow as tf


# num rois means batch size.
class RoiPooling(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPooling, self).__init__(**kwargs)

    # pooling에서 학습할 인자는 따로 없습니다. 채널 수만 따옵니다.
    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]
        super(RoiPooling, self).build(input_shape)

    def call(self, x):
        feature_map = x[0]
        rois = x[1]

        outputs = []
        for i in range(self.num_rois):
            xmin = K.cast(rois[i, 0], 'int32')
            xmax = K.cast(rois[i, 1], 'int32')
            ymin = K.cast(rois[i, 2], 'int32')
            ymax = K.cast(rois[i, 3], 'int32')

            # image resize를 이용한 pooling
            res = tf.image.resize_images(feature_map[:, ymin:ymax, xmin:xmax, :], (self.pool_size, self.pool_size))

            outputs.append(res)

        final_output = K.concatenate(outputs, axis=0)

        return final_output

        # output은 다음과 같은 형태입니다.

    def compute_output_shape(self, input_shape):
        return self.num_rois, self.pool_size, self.pool_size, self.nb_channels


def shared_cnn_model():
    # load pretrained vgg 16
    vgg16 = VGG16(include_top=False, weights='imagenet')

    # remove last maxpooling layer
    return Model(vgg16.inputs, vgg16.layers[-2].output)


# region proposal network
def rpn_model(img_input, shared_model):
    num_anchors = 9

    x = Lambda(lambda x: (x / 127.5) - 1.0)(img_input)  # normalization layer
    x = shared_model(x)

    rpn_conv = Conv2D(512, kernel_size=3, padding='same')(x)

    rpn_cls = Conv2D(num_anchors,
                     kernel_size=1,
                     padding='same',
                     activation='sigmoid')(rpn_conv)

    rpn_regression = Conv2D(num_anchors * 4,
                            kernel_size=1,
                            padding='same',
                            activation='linear')(rpn_conv)

    return Model(img_input, [rpn_cls, rpn_regression])


# classification model
def cls_model(img_input, roi_input, shared_model, NUM_ROIS=32):
    x = Lambda(lambda x: (x / 127.5) - 1.0)(img_input)  # normalization layer
    x = shared_model(x)

    x = RoiPooling(pool_size=7, num_rois=NUM_ROIS)([x, roi_input])

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    # class number include background
    cls = Dense(21, activation='softmax')(x)

    # remove background regr
    regr = Dense(4 * 20, activation='softmax')(x)

    return Model([img_input, roi_input], [cls, regr])
