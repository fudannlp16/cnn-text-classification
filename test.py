# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import cPickle
from model import Model
import tensorlayer as tl
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "/home/liuxiaoyu/PycharmProjects/cnn-text-classification-tf-master/runs/1478434371/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags() #Loading Parameters
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


npzfile=np.load(open("data/testdata.npz","rb"))
x_train=npzfile['arr_0']
y_train=npzfile['arr_1']
x_test=npzfile['arr_0']
y_test=npzfile['arr_1']

# Training
# ==================================================

cnn = Model(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            embedding_size=300,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
saver = tf.train.Saver()
sess=tf.InteractiveSession()

cpkl=tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
print cpkl.model_checkpoint_path
if cpkl and cpkl.model_checkpoint_path:
    saver.restore(sess,cpkl.model_checkpoint_path)

dp_dict=tl.utils.dict_to_one(cnn.outputnetwork.all_drop)
feed_dict={cnn.input_x:x_test,cnn.input_y:y_test}
feed_dict.update(dp_dict)

print  sess.run(cnn.accuracy,feed_dict=feed_dict)


