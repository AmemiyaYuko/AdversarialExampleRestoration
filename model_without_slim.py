import tensorflow as tf
import tensorflow.contrib.layers as layers


def model(is_training=True):
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="images")
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

    conv1 = tf.layers.conv2d(x_input, 256, [3, 3], padding="same",
                             name="conv1", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    conv2 = tf.layers.conv2d(tf.layers.batch_normalization(conv1, training=is_training), 128, [1, 1], padding="same",
                             name="conv2", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    conv3 = tf.layers.conv2d(tf.layers.batch_normalization(conv2, training=is_training), 128, [1, 1], padding="same",
                             name="conv3", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    pool1 = tf.layers.max_pooling2d(conv3, [3, 3], 2, name="pool1")
    drop1 = tf.layers.dropout(pool1, rate=keep_prob, name="drop1")
    conv4 = tf.layers.conv2d(drop1, 256, [3, 3], padding="same",
                             name="conv4", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    conv5 = tf.layers.conv2d(tf.layers.batch_normalization(conv4, training=is_training), 256, [1, 1], padding="same",
                             name="conv5", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    conv6 = tf.layers.conv2d(tf.layers.batch_normalization(conv5, training=is_training), 256, [1, 1], padding="same",
                             name="conv6", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    pool2 = tf.layers.max_pooling2d(conv6, [3, 3], 2, name="pool2")
    drop2 = tf.layers.dropout(pool2, rate=keep_prob, name="drop2")
    conv7 = tf.layers.conv2d(drop2, 256, [3, 3], padding="same",
                             name="conv7", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    conv8 = tf.layers.conv2d(tf.layers.batch_normalization(conv7, training=is_training), 128, [1, 1], padding="same",
                             name="conv8", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    conv9 = tf.layers.conv2d(tf.layers.batch_normalization(conv8, training=is_training), 128, [1, 1], padding="same",
                             name="conv9", kernel_regularizer=layers.l2_regularizer(scale=1e-4))
    pool3 = tf.layers.average_pooling2d(conv9, [2, 2], 2, name="pool3")

    fc1 = tf.layers.dense(layers.flatten(pool3), 512, activation=tf.nn.relu,
                          kernel_regularizer=layers.l2_regularizer(scale=1e-4), name="fc1")
    fc2 = tf.layers.dense(tf.layers.dropout(fc1, rate=keep_prob), 10, name="fc2")
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_input))
    reg = tf.reduce_mean(tf.losses.get_regularization_losses())
    loss = tf.add(loss, reg)
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    lrate = tf.train.exponential_decay(1e-4, step, decay_rate=0.95, decay_steps=1000, staircase=True)
    optimizer = tf.train.AdamOptimizer(lrate).minimize(loss, global_step=step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, 1), tf.argmax(y_input, 1)), dtype=tf.float32))
    prediction = tf.nn.softmax(fc2)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    summary = tf.summary.merge_all()
    return {"loss": loss,
            "accuracy": accuracy,
            "summary": summary,
            "prediction": prediction,
            "optimizer": optimizer,
            "global_step": step,
            "x": x_input,
            "y": y_input,
            "keep_prob": keep_prob}
