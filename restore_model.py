import tensorflow as tf


# using tf 1.4.0 rc
def model(ori_image, adv_image, num_layers=6, image_size=299, is_trainging=False):
    residual_image = tf.add(ori_image, -1.0 * adv_image)
    x = tf.layers.conv2d(inputs=tf.layers.batch_normalization(adv_image, training=is_trainging), filters=128,
                         kernel_size=[1, 1], padding="valid")
    nodes = []
    for i in range(num_layers // 2):
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3], padding="valid",
                             name="conv_%d" % i)
        x = tf.nn.leaky_relu(x, name="lrelu_conv_%d" % i)
        nodes.append(x)
    for i in range(num_layers // 2):
        # skip-connection
        x = tf.concat([x, nodes[num_layers // 2 - i - 1]], axis=0)
        x = tf.layers.conv2d_transpose(inputs=x, filters=128, kernel_size=[3, 3], padding="valid",
                                       name="deconv_%d" % i)
        x = tf.nn.leaky_relu(x, name="lrelu_deconv_%d" % i)
    output = tf.layers.conv2d_transpose(inputs=x, filters=3, kernel_size=[1, 1])
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    lrate = tf.train.exponential_decay(2e-4, step, decay_rate=0.97, decay_steps=2000, staircase=True)
    # loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=output, labels=res_image_flatten)) + tf.reduce_mean(
    # tf.losses.get_regularization_losses())
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.add(residual_image, -1 * output)), reduction_indices=0))
    # + tf.reduce_mean(tf.losses.get_regularization_losses())
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss, global_step=step)
    output_full = tf.add(output, adv_image)
    tf.summary.image("predicted_residual_map", output)
    tf.summary.image("predicted_full_image", output_full)
    tf.summary.scalar("loss", loss)
    summary = tf.summary.merge_all()
    return {"optimizer": optimizer,
            "summary": summary,
            "loss": loss,
            "global_step": step,
            }
