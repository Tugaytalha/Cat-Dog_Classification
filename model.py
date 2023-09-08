import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


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

    def save(self, file):
        self.model.save(file)

    @staticmethod
    def load_image_and_label(filepath, label):
        # Load an image from a filepath and associate it with a label.
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [200, 200])
        img /= 255.0  # normalize to [0,1]
        return img, label

    def plot_roc_curve(self, fpr, tpr, auc):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, matrix, class_names):
        plt.figure(figsize=(4, 4))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def create_dataset(self, filepaths, batch_size, repeat=1, shuffle=1):
        # Create datasets
        list_ds = tf.data.Dataset.list_files(filepaths, shuffle=False)

        # Determine the number of images
        image_count = len(list(list_ds))

        # Convert labels to 0s and 1s
        label_ds = list_ds.map(lambda x: tf.where(tf.strings.regex_full_match(x, ".*\\\\cats\\\\.*"), 0, 1),
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Load and preprocess images
        image_ds = list_ds.map(self.load_and_preprocess_image,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Combine image and label datasets
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        if repeat == 1:
            ds = ds.repeat()

        # Batch, shuffle, and prefetch
        if shuffle == 1:
            ds = ds.shuffle(buffer_size=image_count)
        ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def train_from_array(self, train_files, train_labels, test_files, test_labels, epochs=20, batch_size=64):
        train_dataset = self.create_dataset(train_files, batch_size)
        test_dataset = self.create_dataset(test_files, batch_size)

        train_image_count = len(train_files)
        test_image_count = len(test_files)

        history = self.model.fit(train_dataset, validation_data=test_dataset, epochs=epochs,
                                 steps_per_epoch=train_image_count // batch_size,
                                 validation_steps=test_image_count // batch_size, verbose=0)
        return history

    def evaluate_from_array(self, test_files, test_labels, batch_size=64):
        # Evaluate the model and calculate AUC.
        test_dataset = self.create_dataset(filepaths=test_files, batch_size=batch_size, repeat=0, shuffle=0)
        labels_list = [label for _, label in test_dataset.unbatch().as_numpy_iterator()]
        # Get the true labels and predicted probabilities
        labels_list = np.array(labels_list)
        predicted_probas = self.model.predict(test_dataset)
        """print(predicted_probas)
        predicted_probas = np.array(predicted_probas)
        predicted_probas = predicted_probas * 10000 - 9999"""
        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(labels_list, predicted_probas)
        auc_val = roc_auc_score(labels_list, predicted_probas)
        print(f"AUC: {auc_val:.3f}")

        return auc_val, fpr, tpr, predicted_probas

    @staticmethod
    def decode_img(img, img_size=(200, 200)):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0  # rescale
        return img

    def load_and_preprocess_image(self, path, img_size=(200, 200)):
        img = tf.io.read_file(path)
        return self.decode_img(img, img_size)

    def train(self, train_dir, test_dir, epochs=20, batch_size=64):
        # Create datasets
        train_list_ds = tf.data.Dataset.list_files(str(train_dir + "*/*"), shuffle=False)
        test_list_ds = tf.data.Dataset.list_files(str(test_dir + "*/*"), shuffle=False)

        # Determine the number of train and test images
        train_image_count = len(list(train_list_ds))
        test_image_count = len(list(test_list_ds))

        # Convert labels to 0s and 1s
        train_label_ds = train_list_ds.map(lambda x: tf.where(tf.strings.regex_full_match(x, ".*\\\\cats\\\\.*"), 0, 1),
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_label_ds = test_list_ds.map(lambda x: tf.where(tf.strings.regex_full_match(x, ".*\\\\cats\\\\.*"), 0, 1),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Load and preprocess images
        train_image_ds = train_list_ds.map(self.load_and_preprocess_image,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_image_ds = test_list_ds.map(self.load_and_preprocess_image,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Combine image and label datasets
        train_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
        test_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))
        train_ds = train_ds.repeat()
        test_ds = test_ds.repeat()

        # Batch, shuffle, and prefetch
        train_ds = train_ds.shuffle(buffer_size=train_image_count).batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Train the model
        history = self.model.fit(train_ds, validation_data=test_ds, epochs=epochs,
                                 steps_per_epoch=train_image_count // batch_size,
                                 validation_steps=test_image_count // batch_size, verbose=0)

        return history

    def evaluate(self, test_dir, batch_size=64):
        # create data generator
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        # prepare iterator
        test_it = datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=batch_size,
                                              target_size=(200, 200))

        # evaluate model
        _, acc = self.model.evaluate(test_it, steps=len(test_it), verbose=0)
        return acc * 100.0

    # plot diagnostic learning curves
    @staticmethod
    def summarize_diagnostics(history):
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
        # pyplot.close()


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
