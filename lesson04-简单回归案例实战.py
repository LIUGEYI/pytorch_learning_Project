import numpy as np


# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    # loss = 求和(Wx+b-y)^2
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]  # 点的x值
        y = points[i, 1]  # 点的y值
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))  # Average


def step_gradient(b_current, w_current, points, learningRate):
    # w'= w - lr* dloss/dw
    # dL/dw = 2(wx+b-y)x
    # dL/db = 2(wx++b-y)

    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))  # 除N是做一个AVERAGE
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    # # 沿负梯度方向下降
    new_b = b_current - (learningRate * b_gradient)
    new_m = w_current - (learningRate * w_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # 更新w和b
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )


if __name__ == '__main__':
    run()
