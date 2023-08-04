import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import sys
import tensorflow as tf


class CNNModel:
    def __init__(self):
        self.model = self.define_model()

    def define_model(self):
        # Chose GPU as computing device
        physical_devices = tf.config.list_physical_devices('GPU')
        print(tf.config.list_physical_devices())
        if physical_devices:
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
        else:
            print("No GPU devices found.")
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, test_dir, epochs=20, batch_size=64):
        # create data generator
        datagen = ImageDataGenerator(rescale=1.0/255.0)

        # prepare iterators
        train_it = datagen.flow_from_directory(train_dir, class_mode='binary', batch_size=batch_size,
                                               target_size=(200, 200))
        test_it = datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=batch_size,
                                              target_size=(200, 200))

        # fit model
        history = self.model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
                                           validation_steps=len(test_it), epochs=epochs, verbose=0)
        return history

    def evaluate(self, test_dir, batch_size=64):
        # create data generator
        datagen = ImageDataGenerator(rescale=1.0/255.0)

        # prepare iterator
        test_it = datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=batch_size,
                                              target_size=(200, 200))

        # evaluate model
        _, acc = self.model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
        return acc * 100.0

    # plot diagnostic learning curves
    def summarize_diagnostics(seşf, history):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'], color='blue', label='train')
        pyplot.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['accuracy'], color='blue', label='train')
        pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
        # save plot to file
        filename = sys.argv[0].split('/')[-1]
        pyplot.savefig(filename + '_plot.png')
        pyplot.show()
        #pyplot.close()


def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = CNNModel()
    train_dir = 'dataset_dogs_vs_cats/train/'
    test_dir = 'dataset_dogs_vs_cats/test/'
    epochs = 20
    batch_size = 64

    history = model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size)
    accuracy = model.evaluate(test_dir, batch_size=batch_size)
    model.summarize_diagnostics(history)

    print("Training complete.")
    print("Accuracy: %.3f%%" % accuracy)


if __name__ == '__main__':
    main()
