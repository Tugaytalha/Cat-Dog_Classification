from os import listdir
from numpy import asarray
from numpy import save
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


class Reshaper:
    def __init__(self, dest_folder):
        self.dest_folder = dest_folder

    def process_images(self):
        images, labels = list(), list()

        # iterate each file in destination
        for file in listdir(self.dest_folder):
            # cat or dog labeling
            output = 0.0
            if file.startswith('dog'):
                output = 1.0

            # load image in size of 200x200 pixels. dest + file is name of the image
            # like "train/" + "cat1.jpg" = "train/cat1.jpg"
            image = load_img(self.dest_folder + file, target_size=(200, 200))
            # convert to numpy array
            image = img_to_array(image)
            # store
            images.append(image)
            labels.append(output)

        # convert to a numpy arrays
        images = asarray(images)
        labels = asarray(labels)

        return images, labels

    # save the reshaped images
    @staticmethod
    def save_data(images, labels):
        save('dogs_vs_cats_images.npy', images)
        save('dogs_vs_cats_labels.npy', labels)


def main():
    dest_folder = "train/"
    processor = Reshaper(dest_folder)
    images, labels = processor.process_images()
    print(images.shape, labels.shape)
    processor.save_data(images, labels)


if __name__ == '__main__':
    main()
