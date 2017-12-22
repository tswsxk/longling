# coding: utf-8
# created by tongshiwei on 17-11-5

def textcnn_example():
    from cnnClass import numericalCNN
    from longling.framework.ML.mxnet.nn.shared.text_lib import vecDict

    root = "../../../../../../"
    model_dir = root + "data/text_cnn/test/"
    vecdict = vecDict(root + "data/word2vec/comment.vec.dat")
    nn = numericalCNN(
        sentence_size=25,
        model_dir=model_dir,
        vecdict_info=vecdict.info,
    )
    # nn.network_plot(batch_size=128, show_tag=True)
    # nn.save_model(nn.model_dir + "model.class")
    # nn = NN.load_model(model_dir + "model.class")
    nn.process_fit(
        location_train=root + "data/text/mini.instance.train",
        location_test=root + "data/text/mini.instance.test",
        vecdict=vecdict,
        ctx=-1,
        epoch=20,
    )
    nn.set_predictor(1, -1, 19, vecdict)
    print nn.predict([u"wa ka ka ka ka"])
    nn.clean_predictor()

def numericalcnn_example():
    pass