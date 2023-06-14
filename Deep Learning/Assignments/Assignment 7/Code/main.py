"""
Source code for class-conditioned VAE created in ZhuSuan
Inspired by https://zhusuan.readthedocs.io/en/latest/tutorials/vae.html
Created by: Sahand Sabour (2020280401)
"""

import time
import numpy as np
import zhusuan as zs
import tensorflow as tf
from utils import load_dataset, save_image, shuffle

# Encoder model
@zs.reuse_variables(scope="encoder")
def build_encoder(x, l, z_dim, n_samples):
    bn = zs.BayesianNet()
    x_and_t = tf.concat(values=[x, l], axis=1)
    h = tf.layers.dense(tf.cast(x_and_t, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_samples)

    return bn


# Decoder model
@zs.meta_bayesian_net(scope="decoder", reuse_variables=True)
def build_decoder(x_dim, l_input, z_dim, n, n_samples=1):
    bn = zs.BayesianNet()
    # z ~ N(z|0, I)
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1.0, group_ndims=1, n_samples=n_samples)
    z_and_t = tf.concat(values=[z, tf.expand_dims(l_input, axis=0)], axis=2)
    h = tf.layers.dense(z_and_t, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)

    # For generating images
    # x_logits = f_NN(z)
    x_logits = tf.layers.dense(h, x_dim)
    # x_mean
    bn.deterministic("x_mean", tf.sigmoid(x_logits))
    # x ~ Bernoulli(x|sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1)

    return bn


def run():
    x_train, l_train = load_dataset()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Begin training (inferencing)
        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            x_train, l_train = shuffle(x_train, l_train)
            lbs = []
            for i in range(iter_num):
                x_batch = x_train[i * batch_size : (i + 1) * batch_size]
                l_batch = l_train[i * batch_size : (i + 1) * batch_size]
                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={
                        x_input: x_batch,
                        l_input: l_batch,
                        n_samples: 1,
                        n: batch_size,
                    },
                )
                lbs.append(lb)
            time_epoch += time.time()
            print(
                "Epoch {} ({:.1f}s): Lower bound = {}".format(
                    epoch, time_epoch, np.mean(lbs)
                )
            )

            # Occasionally generate results for each digit 0-9 (for comparison reason)
            if (epoch % 25 == 0) or (epoch == 1):
                labels = np.broadcast_to(np.array([0] * 10), (100, 10)).copy()
                for i in range(10):
                    labels[:, i] = 1
                    images = sess.run(
                        x_gen, feed_dict={l_input: labels, n: 100, n_samples: 1}
                    )
                    name = "results/Epoch{}_Label{}.png".format(epoch, i)
                    save_image(images, name)
                    labels[:, i] = 0


def main():
    # Set of globally used variables
    global x, x_input, x_dim, z_dim, l, l_input, l_dim
    global epochs, batch_size, iter_num, n, n_samples
    global encoder, decoder, lower_bound, is_log_likelihood, infer_op, x_gen

    # Load MNIST
    x_train, l_train = load_dataset()

    # Initialize parameters
    x_dim = x_train.shape[1]
    l_dim = l_train.shape[1]
    z_dim = 40
    epochs = 1000
    batch_size = 128
    iter_num = x_train.shape[0] // batch_size

    # Create necessary placeholders
    n = tf.placeholder(tf.int32, shape=[], name="n")
    n_samples = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random.uniform(tf.shape(x_input)), x_input), tf.int32)
    l_input = tf.placeholder(tf.float32, shape=[None, l_dim], name="l")     
    l = tf.cast(tf.less(tf.random.uniform(tf.shape(l_input)), l_input), tf.int32)

    # Create the encoder and decoder models
    encoder = build_encoder(x, l, z_dim, n_samples)
    decoder = build_decoder(x_dim, l_input, z_dim, n, n_samples)
    # Calculate the lower bound and cost for inference
    lower_bound = zs.variational.elbo(decoder, {"x": x}, variational=encoder, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    # Calculate log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(decoder, {"x": x}, proposal=encoder, axis=0)
    )
    # Inference optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)
    bn = decoder.observe()
    x_gen = tf.reshape(bn["x_mean"], [-1, 28, 28, 1])
    # Start the model training and generation
    run()


if __name__ == "__main__":
    main()
