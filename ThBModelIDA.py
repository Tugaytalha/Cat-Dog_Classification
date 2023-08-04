import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import ThBVGGmodel


class ThreeVGGIDA(ThBVGGmodel.ThreeVGG):

    def __init__(self):
        super().__init__()

    def train(self, train_dir, test_dir, epochs=20, batch_size=64):
        # create data generators
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1,
                                           horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        # prepare iterators
        train_it = train_datagen.flow_from_directory(train_dir, class_mode='binary', batch_size=batch_size,
                                                     target_size=(200, 200))
        test_it = test_datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=batch_size,
                                                   target_size=(200, 200))

        # fit model
        history = self.model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
                                           validation_steps=len(test_it), epochs=epochs, verbose=0)
        return history

    def evaluate(self, test_dir, batch_size=64):
        # create data generator
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        # prepare iterator
        test_it = test_datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=batch_size,
                                                   target_size=(200, 200))

        # evaluate model
        _, acc = self.model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
        return acc * 100.0


def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    new_model = ThreeVGGIDA()
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
