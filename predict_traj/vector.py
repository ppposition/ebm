import numpy as np
import matplotlib.pyplot as plt

# 定义向量场
def vector_field(x, y, alpha=1.0, beta=0.001):
    u = -1 + alpha * x * np.exp(-x**2 - y**2)  # x 方向的速度分量
    #v = np.where(np.abs(y) < 2, beta * y * (np.abs(y) - 2)*(-x*(10-np.abs(x)))*0.02, beta * 0.1 * y * (np.abs(y) - 2))  # 在其他范围，beta 值更小
    v = -beta * y * (-x*(10-np.abs(x)))*0.02
    return u, v

# 欧拉法生成轨迹
def generate_trajectory(x0, y0, step_size=0.05, steps=1000, alpha=1.0, beta=0.5):
    x, y = [x0], [y0]
    for _ in range(steps):
        u, v = vector_field(x[-1], y[-1], alpha, beta)
        x_new = x[-1] + u * step_size
        y_new = y[-1] + v * step_size
        x.append(x_new)
        y.append(y_new)
    trajectory = np.column_stack((np.array(x), np.array(y)))
    return trajectory

# 绘制向量场和轨迹
def plot_vector_field_and_trajectories(ax, x0=None, y0=None, step_size=0.05, alpha=1.0, beta=0.5, steps=100):
    x_vals = np.linspace(-10, 10, 20)
    y_vals = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    U, V = vector_field(X, Y, alpha, beta)

    ax.quiver(X, Y, U, V, color='blue')

    if x0 is not None:
        trajectory= generate_trajectory(x0, y0, step_size, alpha=alpha, beta=beta, steps=steps)
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=f'Trajectory from ({x0:.2f}, {y0:.2f})')

    ax.set_xlim([-10, 10])
    ax.set_ylim([-5, 5])
    ax.set_title(f"Vector Field with Positive and Negative Divergence (alpha={alpha}, beta={beta})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
# 运行

if __name__=='__main__':
    fig, ax = plt.subplots()
    plot_vector_field_and_trajectories(ax, step_size=0.1, alpha=1.0, beta=0.5)
    plt.savefig('vector.png')