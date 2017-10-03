import tensorflow as tf
import tensorflow.contrib.layers as layers


def model():
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="images")
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    get_weights = tf.placeholder(dtype=tf.bool, name="get_weights")
    filters = [128, 256, 384, 512, 864]
    convs = []
    input = x_input
    for i in range(len(filters)):
        convs.append(tf.layers.max_pooling2d(
            tf.layers.conv2d(tf.layers.batch_normalization(input), filters[i], [3, 3], padding="same",
                             name="conv" + str(i)), [2, 2], strides=2, name="pool" + str(i)))
        input = convs[len(convs) - 1]
    # Bottleneck structure
    drop1 = tf.layers.dropout(input, rate=keep_prob)
    bn_conv1 = tf.layers.conv2d(tf.layers.batch_normalization(drop1), 512, [1, 1], padding="valid", name="bn_conv1")
    bn_conv2 = tf.layers.conv2d(tf.layers.batch_normalization(bn_conv1), 256, [1, 1], padding="valid", name="bn_conv2")

    fc1 = tf.layers.dense(layers.flatten(bn_conv2), 512, activation=tf.nn.relu,
                          kernel_regularizer=layers.l2_regularizer(scale=1e-4), name="fc1")
    fc2 = tf.layers.dense(tf.layers.dropout(fc1, rate=keep_prob), 10, name="fc2")
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_input))
    reg = tf.reduce_mean(tf.losses.get_regularization_losses())
    loss = tf.add(loss, reg)
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    lrate = tf.train.exponential_decay(1e-4, step, decay_rate=0.97, decay_steps=3000, staircase=True)
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
            "keep_prob": keep_prob, }
