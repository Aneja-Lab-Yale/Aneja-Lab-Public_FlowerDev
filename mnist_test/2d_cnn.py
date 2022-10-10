import tensorflow as tf

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Toy CNN + Feature Extraction
   Aneja Lab | Yale School of Medicine
   Author: Sanjay Aneja, MD
"""

def CNN_2D(shape):

    inputs = tf.keras.Input(shape=shape)
    x1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name='x1')(inputs)
    x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='x2')(x1)
    x3 = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation="relu", name='x3')(x2)
    x4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='x4')(x3)
    x5 = tf.keras.layers.Flatten(name='x5')(x4)
    x6 = tf.keras.layers.Dropout(0.5, name='x6')(x5)
    x7 = tf.keras.layers.Dense(20, name='x7')(x6)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name='output')(x7)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile("adam", "sparse_categorical_crossentropy", metrics='accuracy')
    return model

