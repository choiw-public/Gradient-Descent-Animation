import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import tensorflow as tf

# use python 2.x.

lrs = [0.1, 0.2, 0.3, 0.35, 0.65, 0.70, 0.75, 0.80, 0.95, 1.0, 1.01, 1.03, 1.05]
x1_x2_range = [-3.0, 3.0]
x_init = 2.0
limit_iteration = 20
convergence = [-0.01, 0.01]

fig = plt.figure(figsize=(6.5, 5))
for lr in list(lrs):
    x_grid = np.linspace(x1_x2_range[0], x1_x2_range[1], 100)
    should_continue = True
    y_true = x_grid ** 2
    x_new = x_init
    trajectory = [[x_new, x_new ** 2]]
    i = 0
    iteration = [i]
    while should_continue:
        i += 1
        x_new = x_new - lr * 2 * x_new
        y = x_new ** 2
        iteration.append(i)
        if convergence[0] <= y <= convergence[1] or i >= limit_iteration:
            should_continue = False
        trajectory.append([x_new, y])

    ax = fig.add_subplot(111)
    ax.plot(x_grid, y_true)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(0, 8.0)
    ax.yaxis.set_ticks(np.arange(0, 8 + 1, 2.0))
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    line, = ax.plot([], [], 'r', lw=2)
    point, = ax.plot([], [], 'ro')
    line.set_data([], [])
    point.set_data([], [])


    def animate(i):
        line.set_data(trajectory[0, :i], trajectory[1, :i])
        point.set_data(trajectory[0, i - 1:i], trajectory[1, i - 1:i])
        ax.title.set_text("alpha: %.2f, iter: %02d, x:%.3f, y: %.3f" % (lr, i, trajectory[0, i], trajectory[1, i]))
        return line, point


    trajectory = np.array(trajectory).T
    anim = FuncAnimation(fig, animate, frames=len(iteration), interval=120, repeat_delay=0, blit=True)

    writer = PillowWriter(fps=2)
    anim.save("gd_function_x2%.2f.gif" % lr, writer=writer)
    fig.clear()
