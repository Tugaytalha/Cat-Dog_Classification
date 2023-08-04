import os
import time
import ThBVGGmodelDout
import ThBModelIDA


def main():
    # INFO and WARNING messages will not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # List of models
    dout_models = [
        ThBVGGmodelDout.ThreeVGGDo(0.0001),
        ThBVGGmodelDout.ThreeVGGDo(0.001),
        ThBVGGmodelDout.ThreeVGGDo(0.003),
        ThBVGGmodelDout.ThreeVGGDo(0.01),
        ThBVGGmodelDout.ThreeVGGDo(0.03)
    ]
    rates = [0.0001, 0.001, 0.003, 0.01, 0.03]

    ida_model = ThBModelIDA.ThreeVGGIDA()
    do_ida_model = ThBModelIDA.ThreeVGGIDA()

    train_dir = 'dataset_dogs_vs_cats/train/'
    test_dir = 'dataset_dogs_vs_cats/test/'
    epochs = 100
    batch_size = 64

    # Train each model and measure the time

    print(f"Training IDA model ...")
    st = time.time()
    # Ensure your model objects have the train method.
    history_ida = ida_model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size)
    print(f"Training complete for IDA model in {time.time() - st} seconds.")
    print(f"Training Drop out IDA model ...")
    st = time.time()
    # Ensure your model objects have the train method.
    history_do_ida = do_ida_model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size)
    print(f"Training complete for Drop out IDA model in {time.time() - st} seconds.")

    dout_histories = []
    # Train each model and measure the time
    for i, model in enumerate(dout_models, 1):
        print(f"Training drop out model with {rates[i-1]} learning rate...")
        st = time.time()

        # Ensure your model objects have the train method.
        dout_histories.append(model.train(train_dir, test_dir, epochs=epochs, batch_size=batch_size))

        print(f"Training complete for drop out model with {rates[i-1]} learning rate in {time.time() - st} seconds.")

    accuracy = ida_model.evaluate(test_dir, batch_size=batch_size)
    print("IDA Model Accuracy: %.3f%%" % accuracy)

    accuracy = do_ida_model.evaluate(test_dir, batch_size=batch_size)
    print("Drop out IDA Model Accuracy: %.3f%%" % accuracy)

    for i, model in enumerate(dout_models, 1):
        accuracy = model.evaluate(test_dir, batch_size=batch_size)
        print("Drop out model with %.4f%% learning rate Accuracy: %.3f%%" % (rates[i-1], accuracy))

    ida_model.summarize_diagnostics(history_ida)
    do_ida_model.summarize_diagnostics(history_do_ida)

    for i, model in enumerate(dout_models, 1):
        model.summarize_diagnostics(dout_histories[i - 1])


if __name__ == '__main__':
    main()
