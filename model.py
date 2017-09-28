import tensorflow as tf
import tensorflow.contrib.slim as slim


def model():
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="images")
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    conv1 = slim.conv2d(x_input, 96, [5, 5], 1, scope="conv1")
    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2], scope="pool1")
    norm1 = slim.batch_norm(pool1, scope="batch_norm1")
    conv2 = slim.conv2d(norm1, 256, [3, 3], 1, scope="conv2")
    pool2 = slim.max_pool2d(conv2, [2, 2], [2, 2], scope="pool2")
    norm2 = slim.batch_norm(pool2, scope="batch_norm2")
    conv3 = slim.conv2d(norm2, 384, [3, 3], 1, scope="conv3")
    conv4 = slim.conv2d(conv3, 384, [3, 3], 1, scope="conv4")
    conv5 = slim.conv2d(conv4, 256, [3, 3], 1, scope="conv5")
    pool5 = slim.batch_norm(slim.max_pool2d(conv5, [2, 2], [2, 2]), scope="conv5")
    flatten = slim.flatten(pool5)
    fc1 = slim.fully_connected(flatten, 512, activation_fn=tf.nn.relu, scope="fc1")
    drop1 = slim.dropout(fc1, keep_prob=keep_prob, scope="drop1")
    norm_fc1 = slim.batch_norm(drop1, scope="fc_norm1")
    fc2 = slim.fully_connected(norm_fc1, 512, activation_fn=tf.nn.relu, scope="fc2")
    drop2 = slim.dropout(fc2, keep_prob=keep_prob, scope="drop2")
    fc3 = slim.fully_connected(drop2, 10, scope="fc3")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y_input))
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc3, 1), tf.argmax(y_input, 1)), dtype=tf.float32))
    prediction = tf.nn.softmax(fc3)
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
