import numpy as np

from spin.operators import adj_kernel, measure_energy, measure_magnetization, measure_heat_capacity


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

    expected_kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]])

    queried_u_kernel = adj_kernel(u_config)
    assert np.all(queried_u_kernel == expected_kernel)

    queried_r_kernel = adj_kernel(r_config)
    assert np.all(queried_r_kernel == expected_kernel)


def test_measure_energy(J=1):
    u_config = uniform_config()
    r_config = random_config()

    expected_u_energy = -9.0 / u_config.size
    queried_u_energy = measure_energy(J, u_config)
    assert queried_u_energy == expected_u_energy

    expected_r_energy = -4.0 / r_config.size
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


def test_measure_heat_capacity():
    j = 1
    t = 1

    u_energy = measure_energy(j, uniform_config())
    r_energy = measure_energy(j, random_config())
    r_energies = np.array([0.14228717, 0.37562505, 0.02317073, 0.73532235, 0.62940131,
                           0.96122942, 0.21199548, 0.92335655, 0.97151338, 0.84795043])

    expected_u_heat_capacity = 0
    queried_u_heat_capacity = measure_heat_capacity(u_energy, t)
    assert queried_u_heat_capacity == expected_u_heat_capacity

    expected_r_heat_capacity = 0
    queried_r_heat_capacity = measure_heat_capacity(r_energy, t)
    assert queried_r_heat_capacity == expected_r_heat_capacity

    expected_r_heat_capacity = 0.11936753280322099
    queried_r_heat_capacity = measure_heat_capacity(r_energies, t)
    assert abs(queried_r_heat_capacity - expected_r_heat_capacity) < 1e-6
