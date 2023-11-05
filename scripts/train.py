import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout

def train(X_train,labels,val_images, val_labels):
    input_layer = keras.Input(shape=(128, 128, 1))

    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    dropout1 = Dropout(0.25)(pool1)  
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(dropout1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    dropout2 = Dropout(0.25)(pool2)  
    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(dropout2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    dropout3 = Dropout(0.25)(pool3)  

    flatten = layers.Flatten()(dropout3)

    dense1 = layers.Dense(128, activation='relu')(flatten)
    dropout4 = Dropout(0.50)(dense1)  
    output_layer = layers.Dense(1, activation='sigmoid')(dropout4)  # Binary classification

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.fit(X_train, labels, batch_size=32,
                    validation_data=(val_images, val_labels),
                    epochs=15,  # Adjust the number of epochs
                    verbose=1)
    