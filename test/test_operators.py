import numpy as np

from spin.operators import adj_kernel, measure_energy, measure_magnetization


def uniform_config():
    return np.ones((3, 3))


def random_config():
    return np.array([[-1, 1, -1, -1],
                     [-1, 1, -1, -1],
                     [-1, 1, -1, -1],
                     [-1, -1, 1, -1]])


def test_adj_kernel():
    u_config = uniform_config()
    r_config = random_config()

    expected_kernel = np.array([0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0])

    queried_u_kernel = adj_kernel(u_config)
    assert queried_u_kernel == expected_kernel

    queried_r_kernel = adj_kernel(r_config)
    assert queried_r_kernel == expected_kernel


def test_measure_energy():
    u_config = uniform_config()
    r_config = random_config()
    J = 1

    expected_u_energy = -9.0
    queried_u_energy = measure_energy(J, u_config)
    assert queried_u_energy == expected_u_energy

    expected_r_energy = -4.0
    queried_r_energy = measure_energy(J, r_config)
    assert queried_r_energy == expected_r_energy


def test_measure_magnetization():
    u_config = uniform_config()
    r_config = random_config()

    expected_u_magnetization = 1.0
    queried_u_magnetization = measure_magnetization(u_config)
    assert queried_u_magnetization == expected_u_magnetization

    expected_r_magnetization = 0.5
    queried_r_magnetization = measure_magnetization(r_config)
    assert queried_r_magnetization == expected_r_magnetization
