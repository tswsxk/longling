# deprecated, due to slow speed
# use batch_scorer
import json
import os
from predict import LstmPredict
import conf

def get_x(m):
    return m['data']

def process_batch_score(location_dat, location_res,\
                 location_vec, model_prefix, epoch, max_cnt=-1):
    print model_prefix

    batch_size = conf.PREDICT_BATCH_SIZE
    params = {
        "num_hidden": conf.NUM_HIDDEN,
        "num_embed": conf.NUM_EMBED,
        "num_label": conf.NUM_LABEL,
        "num_lstm_layer": conf.NUM_LSTM_LAYER,
        "location_vec": location_vec,

        "model_prefix": model_prefix,
        'epoch_num': epoch - 1,
        'idx_gpu': 0,
    }
    scorer = LstmPredict(params)

    f_t = open(location_dat, mode='r')
    f_r = open(location_res, mode='w')
    print "begin give score"
    cnt = 0
    ms = []
    import time
    st = time.time()
    print "-----------------process_score--------------------"
    for line in f_t:
        cnt += 1
        if max_cnt > 0 and cnt > max_cnt:
            break
        if cnt % 10000 == 0:
            et = time.time()
            print "score completed % s, cost %s" % (cnt, et - st)
            st = et
        m = json.loads(line)
        ms.append(m)
        if len(ms) == batch_size:
            sentences = []
            for i in range(batch_size):
                sentences.append(get_x(ms[i]))
            res = [scorer.process(sentence) for sentence in sentences]
            for i in range(batch_size):
                ms[i]['score'] = float(res[i])
            for m in ms:
                line = json.dumps(m, ensure_ascii=False)
                f_r.write(line.encode('utf-8') + '\n')
            ms = []

    sentences = []
    cc = len(ms)
    for i in range(cc):
        sentences.append(get_x(ms[i]))
    res = [scorer.process(sentence) for sentence in sentences]
    for i in range(cc):
        ms[i]['score'] = float(res[i])
    for m in ms:
        line = json.dumps(m, ensure_ascii=False)
        f_r.write(line.encode('utf-8') + '\n')

    f_t.close()
    f_r.close()

