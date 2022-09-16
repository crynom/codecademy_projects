import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.callbacks import EarlyStopping
from keras.metrics import AUC, CategoricalAccuracy
from keras import losses
from keras.optimizer_v2.adam import Adam
from keras import layers

import matplotlib.pyplot as plt
import numpy as np

batch = 32
lr = 0.001

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=.1, height_shift_range=.1, zoom_range=.1)
val_gen = ImageDataGenerator(rescale=1./255)

train_iterator = train_gen.flow_from_directory('./Covid19-dataset/train', color_mode= 'grayscale', batch_size= batch, target_size= (256,256), class_mode= 'categorical')
val_iterator = val_gen.flow_from_directory('./Covid19-dataset/test', color_mode= 'grayscale', batch_size= batch, target_size= (256, 256), class_mode= 'categorical',)


def design_model():
    model = Sequential(name='X-Rays')
    model.add(tf.keras.Input(shape=(256,256,1)))
    model.add(layers.Conv2D(5, 8, strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(4,4), strides=(2,2)))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(3, 6, strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(2, 3, strides=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(.2))
    # model.add(layers.Conv2D(3 , 3, strides=1, activation='relu'))
    # model.add(layers.Dropout(.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr), loss=losses.KLDivergence(), metrics=[AUC(), CategoricalAccuracy()])
    model.summary()
    return model

model = design_model()
es = EarlyStopping(monitor='val_categorical_accuracy', mode='auto', patience=50, verbose=1)

history = model.fit(train_iterator, steps_per_epoch=(train_iterator.samples/batch), epochs=250, validation_data=val_iterator, validation_steps=(val_iterator.samples/batch), callbacks=[es], verbose=1)

#report and confusion matrix
test_steps = np.math.ceil(val_iterator.samples/val_iterator.batch_size)
prediction= model.predict(val_iterator, steps=test_steps)
predicted_classes = np.argmax(prediction, axis=1)
true_classes = val_iterator.classes
class_labels = list(val_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
cm = confusion_matrix(true_classes, predicted_classes)

print(f'''
Confusion Matrix:
{cm}

Classification Report:
{report}''')

#graphs
fig = plt.figure('Evaluation')
fig.suptitle('Loss and Metrics graphs')

ax1 = fig.add_subplot(3,1,1)
ax1.plot(history.history['loss'], label='training')
ax1.plot(history.history['val_loss'], label='validation')
ax1.set_xlabel('# of epochs')
ax1.set_ylabel('loss')
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(3,1,2)
ax2.plot(history.history['auc'], label='training')
ax2.plot(history.history['val_auc'], label='validation')
ax2.set_xlabel('# of epochs')
ax2.set_ylabel('auc')
ax2.legend(loc='upper right')

ax3 = fig.add_subplot(3,1,3)
ax3.plot(history.history['categorical_accuracy'], label='training')
ax3.plot(history.history['val_categorical_accuracy'], label='validation')
ax3.set_xlabel('# of epochs')
ax3.set_ylabel('categorical accuracy')
ax3.legend(loc='upper right')

plt.show()
