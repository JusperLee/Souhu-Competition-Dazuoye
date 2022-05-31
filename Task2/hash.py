# coding: utf-8

import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import shutil
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import pickle as pk
from tqdm import tqdm
if tf.__version__ >= '2.0':
  tf = tf.compat.v1
from tensorflow import feature_column as fc
import comm
import evaluation
import logging
from tensorflow.keras import layers
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1024, 'batch_size')
flags.DEFINE_integer('embed_dim', 32, 'embed_dim')
flags.DEFINE_integer('num_epoch', 1, 'num_epoch')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')

SEED = 2022

ROOT_PATH = "./data"
TRAIN_FILE = os.path.join(ROOT_PATH, "train/sample.csv")
TEST_FILE = os.path.join(ROOT_PATH, "evaluate/sample.csv")



def file_to_dataset(file_path, isTest, shuffle, batch_size, num_epochs):
    if (isTest):
        label_name = None
    else:
        label_name = "label"
    ds = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name=label_name,
        shuffle = shuffle,
        shuffle_seed = SEED,
        num_epochs = num_epochs,
        prefetch_buffer_size = tf.data.experimental.AUTOTUNE,

    )
    return ds



def get_feature_columns():
    feature_columns = list()
    user_cate = fc.categorical_column_with_hash_bucket("suv", 40000)
    feed_cate = fc.categorical_column_with_hash_bucket("itemId", 4000, tf.int64)
    city_cate = fc.categorical_column_with_hash_bucket("city", 200, tf.int64)
    os_cate = fc.categorical_column_with_hash_bucket("osType", 11, tf.int64)
    province_cate = fc.categorical_column_with_hash_bucket("province", 36, tf.int64)
    device_cate = fc.categorical_column_with_hash_bucket("deviceType", 4, tf.int64)
    browser_cate = fc.categorical_column_with_hash_bucket("browserType", 31, tf.int64)
    user_embedding = fc.embedding_column(user_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    feed_embedding = fc.embedding_column(feed_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    city_cate_embedding = fc.embedding_column(city_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    # os_embedding = fc.embedding_column(os_cate, 8, max_norm=FLAGS.embed_l2)
    # province_embedding = fc.embedding_column(province_cate, 8, max_norm=FLAGS.embed_l2)
    # device_embedding = fc.embedding_column(device_cate, 8, max_norm=FLAGS.embed_l2)
    # browser_embedding = fc.embedding_column(browser_cate, 8, max_norm=FLAGS.embed_l2)
    # feature_columns.append(user_embedding)
    # feature_columns.append(feed_embedding)
    # feature_columns.append(city_cate_embedding)
    # feature_columns.append(os_embedding)
    # feature_columns.append(province_embedding)
    # feature_columns.append(device_embedding)
    # feature_columns.append(browser_embedding)

    # emotion_cate = fc.categorical_column_with_hash_bucket("emotion", 100, tf.int64)
    # emotion_embedding = fc.embedding_column(emotion_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    # feature_columns.append(emotion_embedding)

    feature_columns.append(fc.indicator_column(user_cate))
    feature_columns.append(fc.indicator_column(feed_cate))
    feature_columns.append(fc.indicator_column(city_cate))
    return feature_columns


def main():
    with tf.device("/cpu:0"):
        # config = tf.estimator.RunConfig(model_dir=model_dir, tf_random_seed=SEED)

        ds_train = file_to_dataset(TRAIN_FILE, isTest=False, shuffle=False, batch_size=FLAGS.batch_size * 10,
                                    num_epochs=1)

        ds_test = file_to_dataset(TEST_FILE, isTest=False, shuffle=False, batch_size=FLAGS.batch_size * 10, num_epochs=1)
        user_cate = fc.indicator_column(fc.categorical_column_with_hash_bucket("suv", 40000))
        feature_columns = get_feature_columns()
        # print(len(ds_test))
        iterator = list(ds_test.as_numpy_iterator())
        print(len(iterator))
        for i in tqdm(range(len(iterator))):
            x = iterator[i]
            print(len(x))
            print(x[1].shape)
            out = tf.feature_column.input_layer(x[0], feature_columns)
            print(out.shape)
            print(type(out))



if __name__ == "__main__":
    main()

