from Data_Read import tfr_parser
import tensorflow as tf
import tensorflow.keras as keras
from functools import partial

'''
This will go through how to use the data and create a very simple network along with some quick best practices
that will make it easier to read and write networks.
'''

print(tf.__version__)
######################         Best practices          #####################
# add partials to save typing everything out
default_Dense = partial(keras.layers.Dense, activation='relu', kernel_initializer='he_normal')
default_Conv = partial(keras.layers.Conv1D, kernel_size=3, strides = 1, padding='SAME', activation='relu')

# Callbacks, useful for getting the best training result (takes a while to train like this so use when
# you want the best results of a model
early_stopping = keras.callbacks.EarlyStopping(patience=12, restore_best_weights = True)
# stops the model when you start overfitting and gives the best weights back
lr_reduc = keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0, min_lr=8e-6)
# reduces the learning rate if the validation error stops getting better
callbacks = [early_stopping, lr_reduc]


# Define system level variables
dist = "2ft"
path = "D:/UT/Courses/Data_Mining/Final Project/KRI-16Devices-RawData/" + dist + "/"
# path = "/Users/rmd2758/Documents/UT/Courses/Data_Mining/Final Project/KRI-16Devices-RawData/" + dist + "/"
file_name = 'GZIP_2500800_512_2ft_data.tfrecord'
batch_size = 64
validation_size = 0.2

# get the dataset
train, valid, test = tfr_parser(file_name, path, validation_size, batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Input((256, 2)),
    default_Conv(16),
    default_Conv(32),
    keras.layers.Flatten(),
    default_Dense(128),
    default_Dense(16, activation='softmax')
])

optimizer = keras.optimizers.Adam(lr=1e-4) # I like Nadam better than Adam but they are pretty similar
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mae'])
model.summary()

model.fit(train, batch_size=batch_size, epochs=1000, callbacks=callbacks, validation_data=valid)


