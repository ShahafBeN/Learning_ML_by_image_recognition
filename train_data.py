import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

data_dir = 'Marvel_Sanp/data'
image_format = ['jpeg', 'jpg', 'bmp', 'png']
image_classes = []

for image_class in os.listdir(data_dir):
    if image_class[0] == '.':
        continue
    image_classes.append(image_class)

    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path = os.path.join(data_dir, image_class, image)

        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)

            if tip not in image_format:
                print('Could Not Use Image ', image_path)
                os.remove(image_path)
        except Exception as e:
            print(' Problem with Image ', image_path, '\n', e)

data = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                   labels='inferred',
                                                   label_mode='int',
                                                   color_mode='rgb',
                                                   class_names=image_classes,
                                                   image_size=(256, 256),
                                                   batch_size=len(image_classes)+15
                                                   )

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

class_names = data.class_names

# fig, ax = plt.subplots(ncols=len(image_classes))
# for idx, img in enumerate(batch[0][:len(image_classes)]):
#     try:
#         ax[idx].imshow(img.astype(int))
#         ax[idx].title.set_text(batch[1][idx])
#     except Exception as e:
#         print(e)


# Scale Data
data = data.map(lambda x, y: (x/255, y))

# Split Data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# Build Deep Learning Model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(len(image_classes), activation='softmax'))
model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

print(model.summary())

# Train Model

log_dir = 'Marvel_Sanp/Logs/'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
# Save the Model
model.save(os.path.join('Marvel_Sanp/models',
                        'image_classifier_num_1.h5'))

# Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.savefig('Marvel_Sanp/plot_files/loss_val_loss.png')

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.savefig('Marvel_Sanp/plot_files/accuracy_val_accuracy.png')

# Evaluate
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

print(f'precision={precision.result()} \n recall={recall.result()} \n accuracy={accuracy.result()}')




