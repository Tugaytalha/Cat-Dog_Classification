from os import makedirs, listdir
from shutil import copyfile
from random import seed, random


class DatasetOrganizer:
    def __init__(self, dataset_home='dataset_dogs_vs_cats/', val_ratio=0.25):
        self.dataset_home = dataset_home
        self.val_ratio = val_ratio

    def create_directories(self):
        subdirs = ['train/', 'test/']
        for subdir in subdirs:
            # create label subdirectories
            labeldirs = ['dogs/', 'cats/']
            for labldir in labeldirs:
                newdir = self.dataset_home + subdir + labldir
                makedirs(newdir, exist_ok=True)

    def organize_dataset(self, src_directory):
        # seed random number generator
        seed(1)

        # copy training dataset images into subdirectories
        for file in listdir(src_directory):
            src = src_directory + '/' + file
            dst_dir = 'train/'
            # values are between 0 and 1(exclusive), so 0.25 means 25% of the values
            if random() < self.val_ratio:
                dst_dir = 'test/'
            if file.startswith('cat'):
                dst = self.dataset_home + dst_dir + 'cats/' + file
                copyfile(src, dst)
            elif file.startswith('dog'):
                dst = self.dataset_home + dst_dir + 'dogs/' + file
                copyfile(src, dst)


def main():
    dataset_organizer = DatasetOrganizer()
    dataset_organizer.create_directories()
    dataset_organizer.organize_dataset('train/')


if __name__ == '__main__':
    main()
