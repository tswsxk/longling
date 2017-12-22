import conf
from predict import LstmPredict
import json

def eval_data(data_path, scorer):
    count = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with open(data_path) as f:
        for line in f:
            comment = json.loads(line)
            b = scorer.process(comment['x'])
            a = comment['z']
            count += 1
            if b > 0.5 and a == 1:
                tp += 1
            if b > 0.5 and a == 0:
                fp += 1
            if b < 0.5 and a == 1:
                fn += 1
            if b < 0.5 and a == 0:
                tn += 1
            if count % 1000 == 0:
                print count

    p = 1.0 * tp / (tp + fp) if tp + fp else 0
    r = 1.0 * tp / (tp + fn) if tp + fn else 0
    f1 = 2.0 * p * r / (p + r) if p + r else 0
    return float(tp + tn) / count * 100, p * 100, r * 100, f1 * 100


def evaluate(train_path, test_path, location_vec, epoch, model_prefix):
    params = {
        "num_hidden": conf.NUM_HIDDEN,
        "num_embed": conf.NUM_EMBED,
        "num_label": conf.NUM_LABEL,
        "num_lstm_layer": conf.NUM_LSTM_LAYER,
        "location_vec": location_vec,

        "model_prefix": model_prefix,
        'epoch_num': epoch,
        'idx_gpu': 0,
    }
    print "loading model"
    scorer = LstmPredict(params)
    print "load model completed"

    print "[train set] Accuracy: %.3f P:%.3f R:%.3f F:%.3f" % eval_data(train_path, scorer)
    print "[test set] Accuracy: %.3f P:%.3f R:%.3f F:%.3f" % eval_data(test_path, scorer)


if __name__ == "__main__":
    froot = "../../process_comment/"
    root = froot + "rnn_cheat_comment/"
    evaluate(root + "data/rnn_cheat_comment.instance.train", root + "data/rnn_cheat_comment.instance.test", froot + "comment.vec.dat",
             10, root + "model/rnn")


