#coding:utf-8
import math
from text_dnn import *

from text_iterator import TextIdIterator
import logging

def fit(m, text_iter,test_iter, batch_size,\
    optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200, root='model/'):

    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate
    updater = mx.optimizer.get_updater(opt)

    location_log = os.path.join(root, 'model.log')
    location_train_log = os.path.join(root, 'train.log')
    print location_log

    f_log = open(location_log, mode='w')
    train_log = open(location_train_log, 'w')
    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        batch_num = text_iter.cnt/batch_size+ 1
        get_batch_duration = 0.
        real_train_duration = 0.
        # total_loss = 0.
        for _ in range(batch_num):
            train_tic = time.time()
            try:
                tic_ = time.time()
                batchX,batchY = text_iter.next_batch(batch_size)
                get_batch_duration += time.time() - tic_

            except Exception, e:
                logging.error("loading data error")
                logging.error(repr(e))
                continue
        # for batchX, batchY, data_time in text_iter:
        #     get_batch_duration += data_time
            #print 'x',batchX.shape,batchY.shape
            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX 
            m.label[:] = batchY
            m.dnn_exec.forward(is_train=True)
            m.dnn_exec.backward()
            norm = 0
            for idx, weight, grad, name in m.param_blocks:  # param_blocks中没有存储embedding
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= max_grad_norm / norm
                updater(idx, grad, weight)
                grad[:] = 0.0
            real_train_duration += time.time() - train_tic
            # total_loss += cross_entropy(m.dnn_exec.outputs[0].asnumpy(), batchY)
            num_correct += sum(batchY == np.argmax(m.dnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            logging.info('reset learning rate to %g' % opt.lr)

        toc = time.time()
        train_time = toc - tic

        train_acc = num_correct * 100 / float(num_total)
        if (iteration + 1) % 1 == 0:
            m.symbol.save(os.path.join(root, 'dnn-symbol.json'))

            save_dict = {'arg:%s' % k:v for k, v in m.dnn_exec.arg_dict.items() if k != "embedding_weight"}
            save_dict.update({'aux:%s' % k:v for k, v in m.dnn_exec.aux_dict.items() if k != "embedding_weight"})
            param_name = os.path.join(root, 'dnn-%04d.params' % iteration)
            mx.nd.save(param_name, save_dict)
            logging.info('Saved checkpoint to %s' % param_name)

        if (iteration + 1) == epoch:
            save_dict_cpu = {k:v.copyto(mx.cpu()) for k,v in save_dict.items() if k != "embedding_weight"}
            mx.nd.save(param_name+'.cpu',save_dict_cpu)
        num_correct = 0
        num_total = 0
        ps = []
        batch_num = test_iter.cnt/batch_size + 1
        y_dev_batch = []
        for _ in range(batch_num):
            try:
                batchX,batchY = test_iter.next_batch(batch_size)
            except Exception,e:
                logging.error(repr(e))
                continue
            if batchX.shape[0] != batch_size:
                continue
            y_dev_batch.extend(batchY)
            m.data[:] = batchX
            m.dnn_exec.forward(is_train=False)
            ps.extend(np.argmax(m.dnn_exec.outputs[0].asnumpy(), axis=1))
            num_correct += sum(batchY == np.argmax(m.dnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        cat_res = evaluate(y_dev_batch, ps)
        dev_acc = num_correct * 100 / float(num_total)
        
        logging.info('Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc))
        print "It takes %s to get batch, real train time %ss" % (get_batch_duration, real_train_duration)
        logging.info('--- Dev Accuracy thus far: %.3f' % dev_acc)

        line = 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)
 
        f_log.write(line.encode('utf-8')+'\n')
        f_log.flush()

        line = '--- Dev Accuracy thus far: %.3f' % dev_acc
        f_log.write(line.encode('utf-8')+'\n')
        f_log.flush()

        eval_res = {}
        eval_res['train_time'] = train_time 
        eval_res['train_acc'] = train_acc
        eval_res['dev_acc'] = dev_acc
        eval_res['prf'] = {}
        for cat, res in cat_res.items():
            logging.info('--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2]))
            eval_res['prf'][cat] = (res[0], res[1], res[2])
            line = '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
            f_log.write(line.encode('utf-8')+'\n')
            f_log.flush()
        train_log.write(json.dumps(eval_res, ensure_ascii=False).encode('utf8')+'\n')    
        train_log.flush()

    train_log.close()
    f_log.close()


def cross_entropy(score_dist, labels):
    loss = 0.
    for i, idx in enumerate(labels):
        if score_dist[i][idx] == 0.0:
            loss += math.log(1e-9, math.e)
        else:
            loss += math.log(score_dist[i][idx], math.e)
    return -loss / len(labels)
