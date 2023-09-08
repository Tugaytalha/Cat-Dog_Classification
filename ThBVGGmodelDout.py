import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import model


class ThreeVGGDo(model.CNNModel):

    def __init__(self, lr=0.001):
        self.model = self.define_model(lr)

    # override define cnn model
    def define_model(self, lr=0.001):
        # Chose GPU as computing device
        physical_devices = tf.config.list_physical_devices('GPU')
        print(tf.config.list_physical_devices())
        if physical_devices:
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
        else:
            print("No GPU devices found.")
        model_tbd = Sequential()
        # block 1
        model_tbd.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                             input_shape=(200, 200, 3)))
        model_tbd.add(MaxPooling2D((2, 2)))
        model_tbd.add(Dropout(0.2))
        # block 2
        model_tbd.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_tbd.add(MaxPooling2D((2, 2)))
        model_tbd.add(Dropout(0.2))
        # block 3
        model_tbd.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_tbd.add(MaxPooling2D((2, 2)))
        model_tbd.add(Dropout(0.2))
        model_tbd.add(Flatten())
        model_tbd.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model_tbd.add(Dropout(0.5))
        model_tbd.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = SGD(learning_rate=lr, momentum=0.9)
        model_tbd.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model_tbd


def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    new_model = ThreeVGGDo()
    train_dir = 'dataset_dogs_vs_cats/train/'
    test_dir = 'dataset_dogs_vs_cats/test/'
    epochs = 50
    batch_size = 64

    history = new_model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size)
    accuracy = new_model.evaluate(test_dir, batch_size=batch_size)
    new_model.summarize_diagnostics(history)

    print("Training complete.")
    print("Accuracy: %.3f%%" % accuracy)


if __name__ == '__main__':
    main()
