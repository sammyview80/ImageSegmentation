import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input, Conv2DTranspose, concatenate



def conv2d_block(input_tensor, n_filters, kernel_size=3, batchNormal=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchNormal:
        x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchNormal:
        x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def get_unet(input_shape=(96, 96, 3), n_filters=12, dropout=0.5, batchNormal=True):
    input_img = Input((input_shape))
    # Downsample
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchNormal=batchNormal)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout*0.5)(p1)
    
    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchNormal=batchNormal)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchNormal=batchNormal)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchNormal=batchNormal)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchNormal=batchNormal)
    
    # Upsampling
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchNormal=batchNormal)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchNormal=batchNormal)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchNormal=batchNormal)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchNormal=batchNormal)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs])
    return model