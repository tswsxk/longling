# coding: utf-8
# 2021/8/5 @ tongshiwei

def set_embedding(embedding, array):
    embedding.weight.set_data(array)
