# coding: utf-8
# create by tongshiwei on 2018/7/8

from tqdm import tqdm
import gensim
import json

from longling.lib.stream import wf_open, wf_close


def bin2tup(loc_bin, loc_dat):
    bin_dict = gensim.models.KeyedVectors.load_word2vec_format(
        loc_bin,
        binary=True
    )
    wf = wf_open(loc_dat)
    for word in tqdm(bin_dict.index2word):
        print(json.dumps((word, bin_dict[word].tolist()), ensure_ascii=False), file=wf)
    wf_close(wf)


if __name__ == '__main__':
    root = "../../../../../data/vec/"
    bin2tup(
        root + "news_12g_baidubaike_20g_novel_90g_embedding_64.vec.bin",
        root + "news_12g_baidubaike_20g_novel_90g_embedding_64.vec.tup"
    )

