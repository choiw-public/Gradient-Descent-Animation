import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import os
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf

tf.random.set_random_seed(0)

# use python 2.x.
use_bnorm = True
optimizer = 'momentum'  # gd for gradient descent, momentum for gd + momentum
loss_type = 'mse'  # log_cosh or mse
kernel_init = 'glorot'  # glorot, he, or bad

# good setting
x1_x2_range = [-3.0, 3.0]
w1_w2_range = [-1.0, 1.0]
w1_true = 0.5
w2_true = 0.25

max_epoch = 10000
record_interval = 50

# lr_start = 0.01  # gd: 0.25, momentum:0.05
# lr_end = 0.001

# predefined lr
if use_bnorm:
    if optimizer == 'momentum':
        if kernel_init == 'he':
            if loss_type == 'log_cosh':
                lr_start = 0.00002  # 0.00001
                lr_end = 0.000019  # 0.000009
                y_lim = 2.0  # y limit of iteration vs cost graph
            elif loss_type == 'mse':
                lr_start = 0.006
                lr_end = 0.0059
                y_lim = 0.25  # y limit of iteration vs cost graph
            else:
                raise ValueError("Unexpected")
        if kernel_init == 'glorot':
            if loss_type == 'log_cosh':
                lr_start = 0.00001
                lr_end = 0.000009
                y_lim = 2.0  # y limit of iteration vs cost graph
            elif loss_type == 'mse':
                lr_start = 0.003
                lr_end = 0.0029
                y_lim = 0.25  # y limit of iteration vs cost graph
            else:
                raise ValueError("Unexpected")
    elif optimizer == 'gd':
        if kernel_init == 'he':
            if loss_type == 'log_cosh':
                lr_start = 0.00007
                lr_end = 0.000007
                y_lim = 2.0  # y limit of iteration vs cost graph
            elif loss_type == 'mse':
                lr_start = 0.01
                lr_end = 0.001
                y_lim = 0.25  # y limit of iteration vs cost graph
            else:
                raise ValueError("Unexpected")
        else:
            raise ValueError('not configured yet')
else:
    if optimizer == 'momentum':
        if kernel_init == 'he':
            if loss_type == 'log_cosh':
                # this does not work in any case
                lr_start = 0.000001
                lr_end = 0.0000009
                y_lim = 2.0  # y limit of iteration vs cost graph
            elif loss_type == 'mse':
                # this does not work in any case
                lr_start = 0.001
                lr_end = 0.001
                y_lim = 0.25  # y limit of iteration vs cost graph
            else:
                raise ValueError("Unexpected")
        elif kernel_init == 'glorot':
            if loss_type == 'log_cosh':
                raise ValueError('not implemented yet')
                # lr_start = 0.00001
                # lr_end = 0.000009
                # y_lim = 2.0  # y limit of iteration vs cost graph
            elif loss_type == 'mse':
                # this does not work in any case
                lr_start = 0.05
                lr_end = 0.0001
                y_lim = 0.25  # y limit of iteration vs cost graph

            else:
                raise ValueError('not implemented yet')
        elif kernel_init == 'bad':
            if loss_type == 'log_cosh':
                lr_start = 0.000001
                lr_end = 0.0000009
                y_lim = 2.0  # y limit of iteration vs cost graph
        else:
            raise ValueError("Unexpected")

    else:
        raise ValueError('not implemented yet')


def f_x1x2(x1, x2, a, b):
    # df/dw1 = x1^2 * cos(2 * x1 - exp(x2) +1) * cos(w1 * x1^2 - w2 * x2^2 +3)
    # df/dw2 = -x2^2 * cos(2 * x1 - exp(x2) +1) * cos(w1 * x1^2 - w2 * x2^2 +3)
    return np.sin(a * x1 ** 2 - b * x2 ** 2 + 3) * np.cos(2 * x1 + 1 - np.exp(x2))


x_grid = np.linspace(x1_x2_range[0], x1_x2_range[1], 100)
grid_x1, grid_x2 = np.meshgrid(x_grid, x_grid)
z_true = f_x1x2(grid_x1, grid_x2, w1_true, w2_true)

data = {"x1": grid_x1,
        "x2": grid_x2,
        "z": z_true}


def mse(predictions, ground_truths):
    # mean-squared-error
    ground_truths = ground_truths.flatten().astype(np.float32)
    squre = (ground_truths - predictions) ** 2
    return tf.reduce_mean(squre, axis=-1)


def mae(predictions, ground_truths):
    # mean-absolute-error
    ground_truths = ground_truths.flatten().astype(np.float32)
    squre = tf.abs(ground_truths - predictions)
    return tf.reduce_mean(squre, axis=-1)


def log_cosh(predictions, ground_truths):
    # mean-absolute-error
    ground_truths = ground_truths.flatten().astype(np.float32)
    loss = tf.log(tf.cosh(predictions - ground_truths))
    return tf.reduce_sum(loss, axis=-1)


if loss_type == 'mse':
    cost_fn = mse
elif loss_type == 'mae':
    cost_fn = mae
elif loss_type == 'log_cosh':
    cost_fn = log_cosh
else:
    raise ValueError('unexpected')

w_grid = np.linspace(w1_w2_range[0], w1_w2_range[1], 100)
grid_w1, grid_w2 = np.meshgrid(w_grid, w_grid)
pred_surface = f_x1x2(data["x1"].reshape([1, -1]),
                      data["x2"].reshape((1, -1)),
                      grid_w1.reshape([-1, 1]),
                      grid_w2.reshape([-1, 1]))
pred_surface = tf.reshape(tf.convert_to_tensor(pred_surface, tf.float32), [100, 100, -1])
cost_landscape = cost_fn(pred_surface, data["z"])

epoch = tf.placeholder(tf.float32)
lr = lr_start - epoch * (lr_start - lr_end) / max_epoch

if optimizer == 'gd':
    optm = tf.train.GradientDescentOptimizer(lr)
elif optimizer == 'momentum':
    optm = tf.train.MomentumOptimizer(lr, 0.9)
else:
    raise ValueError('unexpected optimizer')


def model(x, training):
    ends = []
    if kernel_init == 'glorot':
        init = tf.initializers.glorot_uniform()
    elif kernel_init == 'he':
        init = tf.initializers.he_uniform()
    elif kernel_init == 'bad':
        init = tf.initializers.random_normal(-4.0, 2.6464)
    else:
        raise ValueError("unexpected kernel_init")
    net = x

    for unit_num in [32, 16, 64, 32, 128, 64, 256]:
        net = tf.layers.dense(net, unit_num, None, not use_bnorm, kernel_initializer=init)
        if use_bnorm:
            net = tf.layers.batch_normalization(net, training=training)
        ends.append(net)
        net = tf.nn.relu(net)

    pred = tf.layers.dense(net, 1, None, False, kernel_initializer=init)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        cost = cost_fn(tf.reshape(pred, [-1]), data["z"])
    return pred, cost, ends


tic = time.time()
is_training = tf.placeholder(tf.bool)
x = tf.convert_to_tensor(np.stack([data["x1"].reshape(-1), data["x2"].reshape(-1)], 1), dtype=tf.float32)
pred, cost, ends = model(x, is_training)
pred_all = []
cost_all = []
epoch_all = []
grads_and_vars = optm.compute_gradients(cost)
train_op = optm.apply_gradients(grads_and_vars)

if use_bnorm:
    tf.summary.histogram("w1-grad", grads_and_vars[0][0])
    tf.summary.histogram("w3-grad", grads_and_vars[6][0])
    tf.summary.histogram("w5-grad", grads_and_vars[12][0])
    tf.summary.histogram("w7-grad", grads_and_vars[18][0])
    tf.summary.histogram("w1", grads_and_vars[0][1])
    tf.summary.histogram("w3", grads_and_vars[6][1])
    tf.summary.histogram("w5", grads_and_vars[12][1])
    tf.summary.histogram("w7", grads_and_vars[18][1])
else:
    tf.summary.histogram("w1-grad", grads_and_vars[0][0])
    tf.summary.histogram("w3-grad", grads_and_vars[2][0])
    tf.summary.histogram("w5-grad", grads_and_vars[4][0])
    tf.summary.histogram("w7-grad", grads_and_vars[6][0])
    tf.summary.histogram("w1", grads_and_vars[0][1])
    tf.summary.histogram("w3", grads_and_vars[2][1])
    tf.summary.histogram("w5", grads_and_vars[4][1])
    tf.summary.histogram("w7", grads_and_vars[6][1])

tf.summary.histogram("layer1", ends[0])
tf.summary.histogram("layer3", ends[2])
tf.summary.histogram("layer5", ends[4])
tf.summary.histogram("layer7", ends[6])

summary_op = tf.summary.merge_all()

if use_bnorm:
    ptn1 = 'with_bnorm'
else:
    ptn1 = 'without_bnorm'

model_name = "./deep_%s_%s_%s_%s" % (ptn1, optimizer, loss_type, kernel_init)
summary_writer = tf.summary.FileWriter(logdir=model_name)


def get_minmax(data, holder):
    if holder:
        minmax = [min(data.min(), min(holder)), max(data.max(), max(holder))]
    else:
        minmax = [data.min(), data.max()]
    return minmax


if not os.path.exists("./record"):
    os.mkdir("./record")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cost_landscape = sess.run(cost_landscape)
    for i in range(max_epoch):
        pred_np = sess.run(pred, feed_dict={is_training: False})
        _, cost_np, ends_np, lr_np = sess.run([train_op, cost, ends, lr], feed_dict={epoch: i, is_training: True})
        if i % record_interval == 0 or i == 0 or i == max_epoch - 1:
            summary_writer.add_summary(sess.run(summary_op, feed_dict={is_training: False}), i)
            pred_all.append(pred_np.reshape(100, 100))
            cost_all.append(cost_np)
            epoch_all.append(i)
            print("epoch: %d, cost: %.8f lr:%.8f" % (i, cost_np, lr_np))


def ax_setup(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


# true x1-x2-f (ground truth)
fig = plt.figure(figsize=(20, 5))
fig_ax1 = fig.add_subplot(131, projection="3d", elev=52.9, azim=57.2)
fig_ax1.plot_surface(data["x1"], data["x2"], data["z"], rstride=3, cstride=3, edgecolor='k', linewidth=0.3, alpha=0.8, cmap=plt.cm.viridis)
ax_setup(fig_ax1)
fig_ax1.set_xlim(x1_x2_range[0], x1_x2_range[1])
fig_ax1.set_ylim(x1_x2_range[0], x1_x2_range[1])
fig_ax1.set_zlim(-1, 1)
fig_ax1.title.set_text("True solution (w1: %.2f, w2:%.2f)" % (w1_true, w2_true))

# predicted x1-x2-f over iteration
fig_ax2 = fig.add_subplot(132, projection='3d', elev=52.9, azim=57.2)
ax_setup(fig_ax2)
fig_ax2.set_xlim(x1_x2_range[0], x1_x2_range[1])
fig_ax2.set_ylim(x1_x2_range[0], x1_x2_range[1])
fig_ax2.set_zlim(-1, 1)
surface = [fig_ax2.plot_surface(grid_x1, grid_x2, pred_all[0], rstride=3, cstride=3, edgecolor='k', linewidth=0.3, alpha=0.8, cmap=plt.cm.viridis)]

# iteratoin vs cost
fig_ax3 = fig.add_subplot(133)
fig_ax3.set_xlim(epoch_all[0], epoch_all[-1])
fig_ax3.set_ylim(0, y_lim)

line, = fig_ax3.plot([], [], 'r', lw=2)
point, = fig_ax3.plot([], [], 'ro')
line.set_data([], [])
point.set_data([], [])


def animate(i):
    iteration = epoch_all[i]
    surface[0].remove()
    surface[0] = fig_ax2.plot_surface(grid_x1, grid_x2, pred_all[i], rstride=3, cstride=3, edgecolor='k', linewidth=0.3, alpha=0.8, cmap=plt.cm.viridis)
    fig_ax2.set_xlim(x1_x2_range[0], x1_x2_range[1])
    fig_ax2.set_ylim(x1_x2_range[0], x1_x2_range[1])
    fig_ax2.set_zlim(-1, 1)
    fig_ax2.title.set_text("Prediction (iter: %05d)" % iteration)

    fig_ax3.title.set_text("Iteration vs cost (iter: %05d, cost:%.4f )" % (iteration, cost_all[i]))
    line.set_data(epoch_all[:i], cost_all[:i])
    point.set_data(epoch_all[i - 1:i], cost_all[i - 1:i])
    return line, point


cost_all = np.array(cost_all).T
anim = FuncAnimation(fig, animate, frames=cost_all.shape[0], interval=60, repeat_delay=0, blit=True)
writer = PillowWriter(fps=15)
anim.save(model_name + ".gif", writer=writer)
toc = time.time() - tic
print("animation generation time: %.4f min" % (toc / 60))
plt.close()
