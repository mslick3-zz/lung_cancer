from keras.layers.convolutional import Convolution2D, UpSampling2D, Convolution3D, MaxPooling3D
from keras.engine.topology import Input
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, merge
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam

def dice_coef(y_true, y_pred):
    """
    Helper method for a custom loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Define a custom loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    return -dice_coef(y_true, y_pred)

def unet_model():
    """
    The UNET model is compiled in this function.
    :return:
    """
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def classifier(input_shape, kernel_size, pool_size):
    """
    Creates a keras model with 3D CNNs and returns the model.
    :param input_shape:
    :param kernel_size:
    :param pool_size:
    :return:
    """
    model = Sequential()

    model.add(Convolution3D(16, kernel_size[0], kernel_size[1], kernel_size[2],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Convolution2D(32, kernel_size[0], kernel_size[1], kernel_size[2]))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Convolution3D(64, kernel_size[0], kernel_size[1], kernel_size[2]))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def train_classifier(input_shape):
    model = classifier(input_shape, (3, 3, 3), (2, 2, 2))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    '''
    Read the preprocessed datafiles chunk by chunnk and train the model on that batch (trainX, trainY) using:'''
    model.train_on_batch(trainX, trainY, sample_weight=None)
    '''The model can be trained on many epochs using for loops'''

    '''
    AFter training the dataset we test our model of the test dataset, read the test file chunk by chunk and
    test on each chunk (trainX, trainY) using:'''
    print(model.test_on_batch(trainX, trainY, sample_weight=None))
    '''The average of all of this can be taken to obtain the final test score.'''

    '''After testing save the model using'''
    model.save('my_model.h5')
