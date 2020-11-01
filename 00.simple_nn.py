import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import tensorflow as tf

# use python 2.x.

tf.enable_eager_execution()

optimizer = 'gd'  # 'gd' for gradient descent, 'momentum' for gd + momentum
good_init = False

x1_x2_range = [-3.0, 3.0]
w1_w2_range = [-1.0, 1.0]
w1_true = 0.5
w2_true = 0.25
max_epoch = 100
lr = 0.05

if good_init:
    w1_init = -0.3
    w2_init = -0.22

else:
    w1_init = -0.40
    w2_init = -0.24


def f_x1x2(x1, x2, w1, w2):
    x1 = tf.cast(x1, tf.float32)
    x2 = tf.cast(x2, tf.float32)
    w1 = tf.cast(w1, tf.float32)
    w2 = tf.cast(w2, tf.float32)
    # df/dw1 = x1^2 * cos(2 * x1 - exp(x2) +1) * cos(w1 * x1^2 - w2 * x2^2 +3)
    # df/dw2 = -x2^2 * cos(2 * x1 - exp(x2) +1) * cos(w1 * x1^2 - w2 * x2^2 +3)
    return tf.sin(w1 * x1 ** 2 - w2 * x2 ** 2 + 3) * tf.cos(2 * x1 + 1 - tf.exp(x2))


x_grid = np.linspace(x1_x2_range[0], x1_x2_range[1], 100)
grid_x1, grid_x2 = np.meshgrid(x_grid, x_grid)
z_true = f_x1x2(grid_x1, grid_x2, w1_true, w2_true)

data = {"x1": grid_x1,
        "x2": grid_x2,
        "z": z_true.numpy()}


def cost_fn(predictions, ground_truths):
    # mean-squared-error
    ground_truths = ground_truths.flatten().astype(np.float32)
    squre = (predictions - ground_truths) ** 2
    return tf.reduce_mean(squre, axis=-1)


w_grid = np.linspace(w1_w2_range[0], w1_w2_range[1], 100)
grid_w1, grid_w2 = np.meshgrid(w_grid, w_grid)
pred_surface = f_x1x2(data["x1"].reshape([1, -1]),
                      data["x2"].reshape((1, -1)),
                      grid_w1.reshape([-1, 1]),
                      grid_w2.reshape([-1, 1]))
pred_surface = tf.reshape(pred_surface, [100, 100, -1])
cost_landscape = cost_fn(pred_surface, data["z"])

# batch gradient descent (whole batch)
w1 = tf.Variable(w1_init, trainable=True, dtype=tf.float32)
w2 = tf.Variable(w2_init, trainable=True, dtype=tf.float32)

if optimizer == "gd":
    optm = tf.train.GradientDescentOptimizer(lr)
elif optimizer == "momentum":
    optm = tf.train.MomentumOptimizer(lr, 0.9)
else:
    raise ValueError('unexpected optimizer')

trajectory = []
for i in range(max_epoch):
    with tf.GradientTape() as tape:
        w1w2 = np.array([w1.numpy(), w2.numpy()]).reshape(-1, 1)
        pred = tf.reshape(f_x1x2(data["x1"], data["x2"], w1, w2), [-1])
        cost = cost_fn(pred, data["z"])
        trajectory.append([w1.numpy(), w2.numpy(), cost.numpy()])
        grads = tape.gradient(cost, [w1, w2])
        optm.apply_gradients(zip(grads, [w1, w2]))


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

trajectory = np.array(trajectory)
pred_all = [f_x1x2(grid_x1, grid_x2, w1_init, w2_init)]
for w1w2z in trajectory:
    w1, w2, z = w1w2z[0], w1w2z[1], w1w2z[2]
    pred_all.append(f_x1x2(grid_x1, grid_x2, w1, w2))

surface = [fig_ax2.plot_surface(grid_x1, grid_x2, pred_all[0], rstride=3, cstride=3, edgecolor='k', linewidth=0.3, alpha=0.8, cmap=plt.cm.viridis)]

# w1-w2-cost
fig_ax3 = fig.add_subplot(133, projection='3d', elev=52.9, azim=57.2)
ax_setup(fig_ax3)
fig_ax3.plot_surface(grid_w1, grid_w2, cost_landscape.numpy(), rstride=3, cstride=3, edgecolor='k', linewidth=0.3, alpha=0.8, cmap=plt.cm.viridis)
fig_ax3.set_xlim(w1_w2_range[0], w1_w2_range[1])
fig_ax3.set_ylim(w1_w2_range[0], w1_w2_range[1])
fig_ax3.set_zlim(0, 1)


del pred_all[0]
line, = fig_ax3.plot([], [], [], 'r', lw=2)
point, = fig_ax3.plot([], [], [], 'ro')
line.set_data([], [])
line.set_3d_properties([])
point.set_data([], [])
point.set_3d_properties([])


def animate(i, trajectory, surface):
    surface[0].remove()
    surface[0] = fig_ax2.plot_surface(grid_x1, grid_x2, pred_all[i], rstride=3, cstride=3, edgecolor='k', linewidth=0.3, alpha=0.8, cmap=plt.cm.viridis)
    fig_ax2.set_xlim(x1_x2_range[0], x1_x2_range[1])
    fig_ax2.set_ylim(x1_x2_range[0], x1_x2_range[1])
    fig_ax2.set_zlim(-1, 1)

    fig_ax2.title.set_text("Prediction (w1: %.2f, w2: %.2f)" % (trajectory[0, i], trajectory[1, i]))
    fig_ax3.title.set_text("iter:%02d, w1: %.2f, w2: %.2f, cost: %.2f" % (i, trajectory[0, i], trajectory[1, i], trajectory[2, i]))
    line.set_data(trajectory[0, :i], trajectory[1, :i])
    line.set_3d_properties(trajectory[2, :i])
    point.set_data(trajectory[0, i - 1:i], trajectory[1, i - 1:i])
    point.set_3d_properties(trajectory[2, i - 1:i])
    return line, point


trajectory = np.array(trajectory).T
anim = FuncAnimation(fig, animate, frames=trajectory.shape[1], fargs=(trajectory, surface), interval=120,
                     repeat_delay=0, blit=True)
if good_init:
    ptn = "good_init"
else:
    ptn = "bad_init"

f = r"./simple_%s_%s.gif" % (optimizer, ptn)
writer = PillowWriter(fps=15)
anim.save(f, writer=writer)
print("done")
#
# if optimizer == 'gd' and not good_init:
#