import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim


def model():
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="images")
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    conv1 = slim.conv2d(x_input, 128, [3, 3], 1, scope="conv1")
    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2], scope="pool1")
    norm1 = slim.batch_norm(pool1, scope="batch_norm1")
    conv2 = slim.conv2d(norm1, 256, [3, 3], 1, scope="conv2")
    pool2 = slim.max_pool2d(conv2, [2, 2], [2, 2], scope="pool2")
    norm2 = slim.batch_norm(pool2, scope="batch_norm2")
    conv3 = slim.conv2d(norm2, 384, [3, 3], 1, scope="conv3")
    conv4 = slim.conv2d(conv3, 128, [1, 1], 1, scope="conv4")
    pool4 = slim.batch_norm(slim.max_pool2d(conv4, [2, 2], [2, 2]), scope="pool4")
    flatten = slim.flatten(pool4)
    fc1 = slim.fully_connected(flatten, 64, activation_fn=tf.nn.relu, scope="fc1",
                               weights_regularizer=layers.l2_regularizer(scale=0.1))
    drop1 = slim.dropout(fc1, keep_prob=keep_prob, scope="drop1")
    fc2 = slim.fully_connected(drop1, 10, scope="fc2", weights_regularizer=layers.l2_regularizer(scale=0.1))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_input))
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=step)
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
