import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.colors as colors
import tensorflow as tf

# use python 2.x.

x1_range = [-2.0, 2.0]
x2_range = [-1.0, 3.0]
x1_init = -0.5
x2_init = 3.0
a = 1
b = 100

limit_iteration = 500
momentum = True  # set 0.0 for gd, otherwise momentum
convergence = [-0.01, 0.01]

fig = plt.figure(figsize=(6.5, 5))


def rosenbrock(x1, x2):
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2


def rosenbrock_dx1(x1, x2):
    return -2 * a + b * x1 * (4 * x1 ** 2 - 4 * x2) + 2 * x1


def rosenbrock_dx2(x1, x2):
    return 2 * b * (x2 - x1 ** 2)


def ax_setup(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


if momentum:
    momentum = 0.9
    ptn = 'momentum'
    lrs = [0.0002]
else:
    momentum = 0
    ptn = 'gd'
    lrs = [0.001]

for lr in list(lrs):
    x1_grid = np.linspace(x1_range[0], x1_range[1], 100)
    x2_grid = np.linspace(x2_range[0], x2_range[1], 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    should_continue = True
    y_true = rosenbrock(x1_mesh, x2_mesh)

    ax = fig.add_subplot(111, projection='3d')
    ax_setup(ax)
    trajectory = [[x1_init, x2_init, rosenbrock(x1_init, x2_init)]]
    x1_old = x1_init
    x2_old = x2_init
    i = 0
    iteration = [i]
    dx1_update_old, dx2_update_old = 0, 0

    while should_continue:
        i += 1
        dx1_update_new = (momentum * dx1_update_old - lr * rosenbrock_dx1(x1_old, x2_old))
        dx2_update_new = (momentum * dx2_update_old - lr * rosenbrock_dx2(x1_old, x2_old))

        x1_new = x1_old + dx1_update_new
        x2_new = x2_old + dx2_update_new
        y = rosenbrock(x1_new, x2_new)
        x1_old, x2_old = x1_new, x2_new
        dx1_update_old = dx1_update_new
        dx2_update_old = dx2_update_new

        iteration.append(i)
        if convergence[0] <= y <= convergence[1] or i >= limit_iteration:
            should_continue = False
        trajectory.append([x1_new, x2_new, y])

    color_norm = colors.LogNorm(vmin=y_true.min(), vmax=y_true.max())
    ax.plot_surface(x1_mesh, x2_mesh, y_true, rstride=3, cstride=3, edgecolor='k', linewidth=0.1, alpha=0.8, norm=color_norm, cmap=plt.cm.viridis)
    ax.set_xlim(x1_range[0], x1_range[1])
    ax.set_ylim(x2_range[0], x2_range[1])
    # ax.set_zlim(y_true.min(), y_true.max())
    # ax.xaxis.set_ticks(np.arange(x1_range[0], x1_range[1], 1.0))
    # ax.yaxis.set_ticks(np.arange(x2_range[0], x2_range[1], 1.0))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    line, = ax.plot([], [], [], 'r', lw=1)
    point, = ax.plot([], [], [], 'ro')
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])


    def animate(i):
        line.set_data(trajectory[0, :i], trajectory[1, :i])
        line.set_3d_properties(trajectory[2, :i])
        point.set_data(trajectory[0, i - 1:i], trajectory[1, i - 1:i])
        point.set_3d_properties(trajectory[2, i - 1:i])
        ax.title.set_text("alpha: %.4f, iter: %02d, x1:%.3f, x2: %.3f, y: %.3f" % (lr, i, trajectory[0, i], trajectory[1, i], trajectory[2, i]))
        return line, point


    trajectory = np.array(trajectory).T
    anim = FuncAnimation(fig, animate, frames=len(iteration), interval=120, repeat_delay=0, blit=True)

    writer = PillowWriter(fps=30)
    anim.save("%s_function_rosenbrock.gif" % ptn, writer=writer)
    fig.clear()
