import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer


# using tf 1.4.0 rc
def model(ori_image, adv_image, num_layers=16, image_size=299, is_trainging=False):
    residual_image = tf.add(ori_image, -1.0 * adv_image)
    x = adv_image  # tf.layers.conv2d(inputs=tf.layers.batch_normalization(adv_image, training=is_trainging), filters=32,
    # kernel_size=[1, 1], padding="valid")
    nodes = []
    nodes.append(x)
    for i in range(num_layers // 2):
        x = tf.layers.batch_normalization(nodes[-1], training=is_trainging)
        x = tf.nn.leaky_relu(x, name="lrelu_conv_%d" % i, alpha=0.2)
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="valid",
                             name="conv_%d" % i, kernel_regularizer=l2_regularizer(0.4))
        nodes.append(x)
    x = nodes[-1]
    for i in range(num_layers // 2):
        x = tf.add(x, nodes[num_layers // 2 - i])
        x = tf.layers.batch_normalization(x, training=is_trainging)
        x = tf.nn.leaky_relu(x, name="lrelu_deconv_%d" % i, alpha=0.2)
        x = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=[3, 3], padding="valid",
                                       name="deconv_%d" % i, kernel_regularizer=l2_regularizer(0.4))

    output = tf.layers.conv2d_transpose(inputs=x, filters=3, kernel_size=[1, 1], padding="valid",
                                        kernel_regularizer=l2_regularizer(0.4))
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    lrate = tf.train.exponential_decay(1e-4, step, decay_rate=0.97, decay_steps=2000, staircase=True)
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=output, labels=residual_image)) + tf.reduce_mean(
        tf.losses.get_regularization_losses())
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.add(residual_image, -1 * output)),
    #                                    reduction_indices=0)) + tf.reduce_sum(tf.losses.get_regularization_losses())

    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss, global_step=step)
    output_full = tf.add(output, adv_image)
    # tf.summary.image("predicted_residual_map", output)
    if step % 50 == 51:
        tf.summary.image("predicted_full_image", output_full)
    tf.summary.scalar("loss", loss)
    summary = tf.summary.merge_all()
    return {"optimizer": optimizer,
            "summary": summary,
            "loss": loss,
            "global_step": step,
            "output": output_full,
            }
