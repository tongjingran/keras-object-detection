from keras.models import Model
from keras import Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Reshape


def yolo_v1():
    img_input = Input((448, 448, 3))
    bbox_input = Input()

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(img_input)
    x = MaxPooling2D()(x)

    x = Conv2D(192, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D()(x)

    for i in range(4):
        x = Conv2D(256, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D()(x)

    for i in range(2):
        x = Conv2D(512, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1470, activation='relu')(x)
    x = Reshape((7, 7, 30))(x)

    model = Model(img_input, x)

    return model


model = yolo_v1()
model.summary()
