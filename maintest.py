import os
import time
import VGG1Dout
import VGG2_2Dout
import ThBVGGmodelDout
import VGG4_4Dout
import VGG5_5Dout
import VGG7_7Dout
import VGG1
import VGG2_2
import ThBVGGmodel
import VGG4_4
import VGG5_5
import VGG7_7


def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # List of models
    dout_models = [
        VGG1Dout.OneVGGDo(),
        VGG2_2Dout.TwoVGGDo(),
        ThBVGGmodelDout.ThreeVGGDo(),
        VGG4_4Dout.FourVGGDo(),
        VGG5_5Dout.FiveVGGDo(),
        VGG7_7Dout.SevenVGGDo()
    ]
    models = [
        VGG1.OneVGG(),
        VGG2_2.TwoVGG(),
        ThBVGGmodel.ThreeVGG(),
        VGG4_4.FourVGG(),
        VGG5_5.FiveVGG(),
        VGG7_7.SeventVGG()
    ]

    train_dir = 'dataset_dogs_vs_cats/train/'
    test_dir = 'dataset_dogs_vs_cats/test/'
    dout_epochs = 50
    epochs = 20
    batch_size = 64

    histories = []
    # Train each model and measure the time
    for i, model in enumerate(models, 1):
        print(f"Training model {i}...")
        st = time.time()

        # Ensure your model objects have the train method.
        histories.append(model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size))

        print(f"Training complete for model {i} in {time.time() - st} seconds.")

    dout_histories = []
    # Train each model and measure the time
    for i, model in enumerate(dout_models, 1):
        print(f"Training drop out model {i}...")
        st = time.time()

        # Ensure your model objects have the train method.
        dout_histories.append(model.train(train_dir, test_dir, epochs=dout_epochs, batch_size=batch_size))

        print(f"Training complete for drop out model {i} in {time.time() - st} seconds.")

    for i, model in enumerate(models, 1):
        accuracy = model.evaluate(test_dir, batch_size=batch_size)
        print("Model %d Accuracy: %.3f%%" % (i, accuracy))

    for i, model in enumerate(dout_models, 1):
        accuracy = model.evaluate(test_dir, batch_size=batch_size)
        print("Drop out model %d Accuracy: %.3f%%" % (i, accuracy))

    for i, model in enumerate(models, 1):
        model.summarize_diagnostics(histories[i-1])

    for i, model in enumerate(dout_models, 1):
        model.summarize_diagnostics(dout_histories[i-1])


if __name__ == '__main__':
    main()
