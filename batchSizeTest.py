import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import matplotlib.pyplot as plt
import VGG7_7


def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    train_dir = 'dataset_dogs_vs_cats/train/'
    test_dir = 'dataset_dogs_vs_cats/test/'
    epochs = 20
    all_filepaths = []
    labels = []

    for category in ['cats', 'dogs']:
        class_dir = os.path.join(train_dir, category)
        for file in os.listdir(class_dir):
            all_filepaths.append(os.path.join(class_dir, file))
            labels.append(0 if category == 'cats' else 1)

    all_filepaths = np.array(all_filepaths)
    labels = np.array(labels)

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    batch_sizes = [8, 16, 32, 64, 128, 256]

    fpr_arr = []
    tpr_arr = []
    auc_arr = []
    cm_arr = []
    # Train each model and measure the time
    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")

        fold_num = 1
        tt = 0
        total_auc = 0
        for train_index, val_index in kf.split(X=np.zeros(len(labels)), y=labels):
            train_files, val_files = all_filepaths[train_index], all_filepaths[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]

            print(f"Training on Fold: {fold_num}")

            model = VGG7_7.SeventVGG()
            st = time.time()
            model.train_from_array(train_files, train_labels, val_files, val_labels, epochs=epochs,
                                   batch_size=batch_size)
            tt += time.time() - st

            auc, fpr, tpr, predicted_probas = model.evaluate_from_array(val_files, val_labels, batch_size)
            accuracy = model.evaluate(test_dir, batch_size=batch_size)
            print("Accuracy: %.3f%%" % accuracy)
            predicted_labels = (predicted_probas > 0.5).astype(int)
            cm = tf.math.confusion_matrix(val_labels, predicted_labels).numpy()
            cm_arr.append(cm)
            total_auc += auc
            fpr_arr.append(fpr)
            tpr_arr.append(tpr)
            auc_arr.append(auc)

            fold_num += 1

        print(f"Training complete for batch size: {batch_size}")
        print(f"Average AUC with batch size {batch_size}: {total_auc / (fold_num - 1):.3f}")
        print(f"Average running time with batch size {batch_size}: {tt / (fold_num - 1):.3f} seconds.")

        # Plot roc curve
    for i in range(len(fpr_arr)):
        print(
            f"PROC curve is plotting for batch size {batch_sizes[i // (fold_num - 1)]}, fold num {i % (fold_num - 1) + 1}")
        model.plot_confusion_matrix(cm_arr[i], ['cats', 'dogs'])
        model.plot_roc_curve(fpr_arr[i], tpr_arr[i], auc_arr[i])


if __name__ == '__main__':
    main()
