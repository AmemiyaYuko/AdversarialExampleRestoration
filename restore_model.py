import tensorflow as tf


# using tf 1.4.0 rc
def model(num_layers=10, image_size=299, is_trainging=False):
    ori_image = tf.placeholder(tf.float16, shape=[None, image_size, image_size, 3])
    adv_image = tf.placeholder(tf.float16, shape=[None, image_size, image_size, 3])
    residual_image = tf.add(ori_image, -1.0 * adv_image)
    x = tf.layers.conv2d(inputs=tf.layers.batch_normalization(adv_image, training=is_trainging), filters=128,
                         kernel_size=[1, 1], padding="valid", kernel_regularizer=tf.nn.l2_loss)
    nodes = []
    for i in range(num_layers // 2):
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3], padding="valid",
                             kernel_regularizer=tf.nn.l2_loss, name="conv_%2d" % i)
        x = tf.nn.leaky_relu(x, name="lrelu_conv_%2d" % i)
        nodes.append(x)
    for i in range(num_layers // 2):
        # skip-connection
        x = tf.stack([x, nodes[num_layers // 2 - i]])
        x = tf.layers.conv2d_transpose(inputs=x, filters=128, kernel_size=[3, 3], padding="valid",
                                       kernel_regularizer=tf.nn.l2_loss, name="deconv_%2d" % i)
        x = tf.nn.leaky_relu(x, name="lrelu_deconv_%2d" % i)
    flatten = tf.layers.flatten(x, name="flatten")
    output = tf.layers.dense(inputs=flatten, units=image_size * image_size, activation=tf.nn.tanh, name="fc",
                             kernel_regularizer=tf.nn.l2_loss)
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    lrate = tf.train.exponential_decay(2e-4, step, decay_rate=0.97, decay_steps=2000, staircase=True)
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=output, labels=residual_image)) + tf.reduce_mean(
        tf.losses.get_regularization_losses())
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss, global_step=step)
    output_full = tf.add(output, tf.layers.flatten(adv_image))
    pictures_full = tf.reshape(output_full, shape=[None, image_size, image_size, 3], name="full_pictures")
    pictures_res = tf.reshape(output, shape=[None, image_size, image_size, 3], name="residual_maps")
    tf.summary.image("predicted_residual_map", pictures_res)
    tf.summary.image("predicted_full_image", pictures_full)
    tf.summary.scalar("loss", loss)
    summary = tf.summary.merge_all()
    return {"optimizer": optimizer,
            "summary": summary,
            "loss": loss,
            "global_step": step,
            "x": ori_image,
            "y": adv_image
            }
