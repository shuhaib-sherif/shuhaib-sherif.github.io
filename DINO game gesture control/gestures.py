import tensorflow as tf
from tensorflow import keras
import os
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Flatten,Dense,Activation
from keras.models import Model
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



#building a model


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(224,224,1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation="relu"),
                                    tf.keras.layers.Dense(2,activation="softmax")])
model.compile(optimizer=tf.optimizers.Adam(),loss="categorical_crossentropy",metrics="accuracy")

#training


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'Datasets/Train',  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 150x150
        batch_size=20,
        color_mode="grayscale",
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')


test_data=ImageDataGenerator(rescale=1/255)
test_generator=test_data.flow_from_directory(
    "Datasets/Test",
    target_size=(224,224),
    batch_size=20,
    color_mode="grayscale",
    class_mode="categorical"
)

history=model.fit(train_generator,
                  steps_per_epoch=5,
                  epochs=5,
                  verbose=1,
                  validation_data=test_generator,
                  validation_steps=8
                  )



model.save('model_hands1.h5')



# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
