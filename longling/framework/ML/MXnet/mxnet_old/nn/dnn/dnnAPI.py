# coding: utf-8
# created by tongshiwei on 17-11-8

def numerical_dnn_example():
    from dnnClass.numericalDNN import numericalDNN

    root = ""
    model_dir = root + "data/numerical_dnn/test/"

    nn = numericalDNN(
        feature_num=28 * 28,
        model_dir=model_dir,
        num_label=10,
        num_hiddens=[1024, 512, 128, 64, 32],
    )
    # nn.save_model(nn.model_dir + "model.class")
    # nn = NN.load_model(model_dir + "model.class")
    nn.network_plot(batch_size=128, show_tag=True)
    # nn.process_fit(
    #     location_train=root + "data/image/one_dim/mnist_train",
    #     location_test=root + "data/image/one_dim/mnist_test",
    #     ctx=-1,
    #     epoch=20,
    #     parameters={'batch_size': 128}
    # )
