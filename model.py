# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
class Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, dropout_keep_prob,l2_reg_lambda=0.0,hidden_unit=64):
        """
        sequence_length – The length of our sentences. Remember that we padded all our sentences to have the same length (59 for our data set).
        num_classes – Number of classes in the output layer, two in our case (positive and negative).
        vocab_size – The size of our vocabulary. This is needed to define the size of our embedding layer, which will have shape [vocabulary_size, embedding_size].
        embedding_size – The dimensionality of our embeddings.
        filter_sizes – The number of words we want our convolutional filters to cover. We will have num_filters for each size specified here. For example, [3, 4, 5] means that we will have filters that slide over 3, 4 and 5 words respectively, for a total of 3 * num_filters filters.
        num_filters – The number of filters per filter size (see above).
        dropout_keep_prob – The keep_prob of DropoutLayer
       """
        self.input_x=tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")
        # self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.dropout_keep_prob=dropout_keep_prob
        self.pad=tf.placeholder(tf.float32,[None,1,embedding_size,1],name="pad")
        # Keeping track of l2 regularization loss (optional)
        l2_loss=tf.constant(0.0)


        network=tl.layers.InputLayer(inputs=self.input_x,
                                     name="input")

        self.padnetwork=tl.layers.InputLayer(inputs=self.pad,
                                        name='padlayer')
        #Inserts a dimension of 1 into a tensor's shape
        network.outputs=tf.expand_dims(network.outputs,-1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_network = []
        sl=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                sl.append(sequence_length-filter_size+1)
                networks=[network]
                networks.extend([self.padnetwork]*(filter_size-1))
                filternetwork=tl.layers.ConcatLayer(layer=networks,concat_dim=1,name="pad-%s" % filter_size)

                #Convolution Layer
                convnetwork=tl.layers.Conv2dLayer(layer=filternetwork,
                                                  act=tf.nn.relu,
                                                  shape=[filter_size,embedding_size,1,num_filters],
                                                  strides=[1,1,1,1],
                                                  padding="VALID",
                                                  W_init=tf.truncated_normal_initializer(stddev=0.1),
                                                  b_init=tf.constant_initializer(0.1),
                                                  name="conv-%s" % filter_size)
                #Maxpooling over the outputs
                poolnetwork=tl.layers.PoolLayer(layer=convnetwork,
                                                ksize=[1,1,1,1],
                                                strides=[1,1,1,1],
                                                padding="VALID",
                                                pool=tf.nn.max_pool,
                                                name="pool-%s" % filter_size
                                                )
                poolnetwork2=tl.layers.ReshapeLayer(poolnetwork,shape=[-1,sequence_length,num_filters],name="poolreshape-%s" % filter_size)
                pooled_outputs_network.append(poolnetwork2)

        # Combine all the pooled features
        self.h_pool_network=tl.layers.ConcatLayer(layer=pooled_outputs_network,concat_dim=2)



        self.grunetwork=tl.layers.RNNLayer(
            layer=self.h_pool_network,
            cell_fn=tf.nn.rnn_cell.GRUCell,
            cell_init_args={},
            n_hidden=hidden_unit,
            return_last=True,
            return_seq_2d=True
        )

        # Add dropout
        self.dropoutnetwork=tl.layers.DropoutLayer(layer=self.grunetwork,keep=self.dropout_keep_prob,name="dropout")
        #

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.outputnetwork=tl.layers.DenseLayer(layer=self.dropoutnetwork,
                                                    n_units=num_classes,
                                                    act=tf.identity,
                                                    b_init=tf.constant_initializer(0.1),
                                                    name="scores"
                                                    )
            self.scores=self.outputnetwork.outputs
            self.predictions=tf.argmax(self.scores,1,name="predictions")


        l2_loss+=tf.nn.l2_loss(self.outputnetwork.all_params[-2])
        l2_loss+=tf.nn.l2_loss(self.outputnetwork.all_params[-1])
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.scores,self.input_y)
            self.loss=tf.reduce_mean(losses)+tf.minimum(l2_reg_lambda*l2_loss,3)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


