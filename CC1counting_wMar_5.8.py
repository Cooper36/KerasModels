from tensorflow.python import keras

from keras import models
from keras import layers
# need slurm
# Keras with data augmentation

model = models.Sequential()
model.add(layers.Conv2D(32, (5,5), padding="same", activation='relu', input_shape=(46,46,3)))
model.add(layers.Conv2D(32, (5,5), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (5,5), padding="same", activation='relu'))
model.add(layers.Conv2D(64, (5,5), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (5,5), padding="same", activation='relu'))
model.add(layers.Conv2D(128, (5,5), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

from keras import optimizers

model.compile(loss='binary_crossentropy',
  optimizer=optimizers.RMSprop(lr=1e-5),
  metrics=['acc'])

import os
#change paths
train_dir = 'GammaCorrected0_75/train'
validation_dir = 'GammaCorrected0_75/validation'
test_dir = 'GammaCorrected0_75/test'

train_CC1neg_dir = os.path.join(train_dir, 'CC1neg')
train_CC1pos_dir = os.path.join(train_dir, 'CC1pos')
train_size = len(os.listdir(train_CC1pos_dir)) + len(os.listdir(train_CC1neg_dir))

analysis_name = 'CC1counting_wMar_5.8'

batch_size = 200
epochs = 100

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=90,
  horizontal_flip=True,
  fill_mode='nearest')
#check out rescale and what it does
validation_datagen = ImageDataGenerator(
  rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  #Need to crop all to same size, add smaller
  target_size=(46,46),
  batch_size=batch_size,
  class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
  validation_dir,
  target_size=(46,46),
  batch_size=batch_size,
  class_mode='binary')

history = model.fit_generator(
  train_generator,
  steps_per_epoch=int(5*train_size/batch_size), # oversample 5x
  epochs=epochs,
  validation_data=validation_generator,
  validation_steps=50)

model.save(analysis_name + '.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# output results as csv
import csv
filename = 'results_' + analysis_name + '.csv'
with open(filename,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['acc','val_acc','loss','val_loss'])
    for i in range(len(history.history['acc'])):
        w.writerow([acc[i], val_acc[i], loss[i], val_loss[i]])

# Evaluate model
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
  test_dir,
  target_size=(46,46),
  batch_size=20,
  class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
