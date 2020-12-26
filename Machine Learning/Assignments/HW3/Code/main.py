import os
import time
import numpy as np
import zhusuan as zs
import tensorflow as tf
from utils import load_dataset

# Encoder (q-net) model
@zs.reuse_variables(scope="encoder")
def build_encoder(x, z_dim, n_z_per_x, std_noise=0):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.cast(x, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    # z_logstd = tf.layers.dense(h, z_dim)+std_noise
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_x)
    return bn


# Decoder model
@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_decoder(x_dim, z_dim, n, n_particles=1):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1.0, group_ndims=1, n_samples=n_particles)
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    # For generating images
    x_logits = tf.layers.dense(h, x_dim)
    bn.deterministic("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


# def gradient_descent(epochs):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         for epoch in range(1, epochs + 1):
#             time_epoch = -time.time()
#             np.random.shuffle(train_image)
#             lbs = []
#             for t in range(iters):
#                 x_batch = train_image[t * batch_size : (t + 1) * batch_size]
#                 _, lb = sess.run(
#                     [infer_op, lower_bound],
#                     feed_dict={x_input: x_batch, n_particles: 1, n: batch_size},
#                 )
#                 lbs.append(lb)
#             time_epoch += time.time()
#             print(
#                 "Epoch {} ({:.1f}s): Lower bound = {}".format(
#                     epoch, time_epoch, np.mean(lbs)
#                 )
#             )

#             if epoch % save_freq == 0:
#                 images = sess.run(x_gen, feed_dict={n: 100, n_particles: 1})
#                 name = os.path.join(result_path, "vae.epoch.{}.png".format(epoch))
#                 save_image_collections(images, name)


if __name__ == "__main__":
    train_image, train_label, test_image, test_label = load_dataset()
    print("loaded dataset")

    x_dim = train_image.shape[1]
    z_dim = 40
    n = tf.compat.v1.placeholder(tf.int32, shape=[], name="n")
    n_particles = tf.compat.v1.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random.uniform(tf.shape(x_input)), x_input), tf.int32)

    gen = build_decoder(x_dim, z_dim, n, n_particles)
    q_net = build_encoder(x_dim, z_dim, n_particles)
    lower_bound = zs.variational.elbo(gen, {"x": x}, variational=q_net, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(gen, {"x": x}, proposal=q_net, axis=0)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)
    
    epochs = 1000
    batch_size = 128
    iters = train_image.shape[0] // batch_size
    test_freq = 100
    test_batch_size = 400
    test_iters = test_image.shape[0] // test_batch_size
    result_path = "results/vae_digits"
    checkpoints_path = "checkpoints/vae_digits"

    # used to save checkpoints during training
    saver = tf.train.Saver(max_to_keep=10)
    save_model_freq = 100

    # run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore the model parameters from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(checkpoints_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        # begin training
        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(train_image)
            lbs = []
            for t in range(iters):
                x_batch = train_image[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            n_particles: 1,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            # test marginal log likelihood
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs, test_lls = [], []
                for t in range(test_iters):
                    test_x_batch = test_image[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1,
                                                  n: test_batch_size})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1000,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test lower bound = {}".format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            # save model parameters
            if epoch % save_model_freq == 0:
                print('Saving model...')
                save_path = os.path.join(checkpoints_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')

        # random generation of images from latent distribution
        x_gen = tf.reshape(gen.observe()["x_mean"], [-1, 28, 28, 1])
        images = sess.run(x_gen, feed_dict={n: 100, n_particles: 1})
        name = os.path.join(result_path, "random_samples.png")
        # save_image_collections(images, name)

        # the following code generates 100 samples for each number
        test_n = [3, 2, 1, 90, 95, 23, 11, 0, 84, 7]
        # map each digit to a corresponding sample from the test set so we can generate similar digits
        for i in range(len(test_n)):
            # get latent distribution from the variational giving as input a fixed sample from the dataset
            z = q_net.observe(x=np.expand_dims(test_image[test_n[i]], 0))['z']
            # run the computation graph adding noise to computed variance to get different output samples
            latent = sess.run(z, feed_dict={x_input: np.expand_dims(test_image[test_n[i]], 0),
                                            n: 1,
                                            n_particles: 100})
            # get the image from the model giving as input the latent distribution z
            x_gen = tf.reshape(gen.observe(z=latent)["x_mean"], [-1, 28, 28, 1])
            images = sess.run(x_gen, feed_dict={})
            name = os.path.join(result_path, "{}.png".format(i))
            # save_image_collections(images, name)