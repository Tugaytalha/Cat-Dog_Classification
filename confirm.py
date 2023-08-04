
from numpy import load
photos = load('dogs_vs_cats_images.npy')
labels = load('dogs_vs_cats_labels.npy')
print(photos.shape, labels.shape)