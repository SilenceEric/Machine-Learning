from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import fill_between

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)