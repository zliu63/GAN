"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.variable_scope(tf.get_variable_scope()): 
            self.Discriminator_Optimizer = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=d_vars)
            self.Generator_Optimizer = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=g_vars)
        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            y = None
            layer3_weights = tf.Variable(tf.random_normal([self._ndims,128]), name = 'd_layer3_weights')
            layer3_bias = tf.Variable(tf.zeros([128]), name = 'd_layer3_bias')
            layer4_weights = tf.Variable(tf.random_normal([128,1]), name = 'd_layer4_weights')
            layer4_bias = tf.Variable(tf.zeros([1]), name = 'd_layer4_bias')
            hidden_layer3 = tf.nn.relu(tf.matmul(x, layer3_weights) + layer3_bias)
            y = tf.matmul(hidden_layer3, layer4_weights) + layer4_bias
            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=tf.ones_like(y)))
        l_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.zeros_like(y_hat)))
        return l_real+l_fake


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            x_hat = None
            layer1_weights = tf.Variable(tf.random_normal([self._nlatent, 128]), name='g_layer1_weights')
            layer1_bias = tf.Variable(tf.zeros(shape=[128]), name='g_layer1_bias')
            layer2_weights = tf.Variable(tf.random_normal([128, self._ndims]), name='g_layer2_bias')
            layer2_bias = tf.Variable(tf.zeros(shape=[self._ndims]), name='g_layer2_bias')
            hidden_layer1 = tf.nn.relu(tf.matmul(z, layer1_weights) + layer1_bias)
            x_hat = tf.matmul(hidden_layer1, layer2_weights) + layer2_bias
            x_hat = tf.nn.sigmoid(x_hat)
            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.zeros_like(y_hat)))
        return l
