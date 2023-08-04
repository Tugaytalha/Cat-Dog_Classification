from matplotlib import pyplot
from matplotlib.image import imread


dest = "train/"
pyplot.figure(figsize=(17, 9))
for i in range(16):

    pyplot.subplot(4, 4, 1 + i)

    filename = dest + 'cat.' + str(i) + '.jpg'

    image = imread(filename)

    pyplot.imshow(image)

pyplot.show()
