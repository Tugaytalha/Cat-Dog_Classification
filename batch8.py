import os
import time
import VGG7_7
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [200, 200])
    img /= 255.0  # Normalize to [0,1]
    return img

def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    new_model = VGG7_7.SeventVGG()
    train_dir = 'dataset_dogs_vs_cats/train/'
    test_dir = 'dataset_dogs_vs_cats/test/'
    epochs = 10
    batch_size = 64
    # Paths
    cat_files = tf.data.Dataset.list_files(test_dir + "cats/*.jpg", shuffle=False)
    dog_files = tf.data.Dataset.list_files(test_dir + "dogs/*.jpg", shuffle=False)

    # Labels
    cat_labels = tf.data.Dataset.from_tensor_slices([0] * len(cat_files))
    dog_labels = tf.data.Dataset.from_tensor_slices([1] * len(dog_files))

    cat_images = cat_files.map(process_path)
    dog_images = dog_files.map(process_path)
    cat_dataset = tf.data.Dataset.zip((cat_images, cat_labels))
    dog_dataset = tf.data.Dataset.zip((dog_images, dog_labels))
    merged_dataset = cat_dataset.concatenate(dog_dataset)
    BATCH_SIZE = 32
    dataset = merged_dataset.batch(BATCH_SIZE)

    st = time.time()
    history = new_model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size)
    print(f"Training complete in {time.time() - st} seconds.")
    accuracy = new_model.evaluate(test_dir, batch_size=batch_size)
    print("Accuracy: %.3f%%" % accuracy)
    predicted_probas = new_model.model.predict(dataset)
    predicted_labels = (predicted_probas > 0.5).astype(int)
    true_labels = np.array([0] * len(cat_files) + [1] * len(dog_files))
    fpr, tpr, _ = roc_curve(true_labels, predicted_probas)
    auc_val = roc_auc_score(true_labels, predicted_probas.squeeze())
    print(auc_val," auc")
    #new_model.save('saved_model/my_model')
    plot_roc_curve(fpr, tpr, auc_val)
    cm = tf.math.confusion_matrix(true_labels, predicted_labels).numpy()
    new_model.plot_confusion_matrix(cm, ['dogs', 'cats'])
    new_model.summarize_diagnostics(history)


if __name__ == '__main__':
    main()