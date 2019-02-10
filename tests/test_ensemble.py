import numpy as np

from spin.ensemble import flip_spin, acceptance_criterion, mc_step, check_convergence, \
    check_autocorrelation, run_mcmc


def uniform_config():
    return np.ones((3, 3))


def random_config():
    return np.array([[-1, 1, -1, -1],
                     [-1, 1, -1, -1],
                     [-1, 1, -1, -1],
                     [-1, -1, 1, -1]])


def test_flip_spin_uniform(seed=42):
    config = uniform_config()

    expected_flip = np.array([[1., 1., 1.],
                              [1., 1., 1.],
                              [-1., 1., 1.]])
    queried_flip = flip_spin(config, seed)
    assert np.all(queried_flip == expected_flip)


def test_flip_spin_random(seed=42):
    config = random_config()

    expected_flip = np.array([[-1, 1, -1, -1],
                              [-1, 1, -1, -1],
                              [-1, 1, -1, 1],
                              [-1, -1, 1, -1]])
    queried_flip = flip_spin(config, seed)
    assert np.all(queried_flip == expected_flip)


def test_acceptance_criterion_low_T(e_i=-4, low_T=1, seed=42):
    lower_energy = -5
    higher_energy = -3

    assert acceptance_criterion(e_i, lower_energy, low_T, seed) == True
    assert acceptance_criterion(e_i, higher_energy, low_T, seed) == None


def test_acceptance_criterion_high_T(e_i=-4, high_T=3, seed=42):
    lower_energy = -5
    higher_energy = -3

    assert acceptance_criterion(e_i, lower_energy, high_T, seed) == True
    assert acceptance_criterion(e_i, higher_energy, high_T, seed) == True


def test_mc_step_uniform_low_T(J=1, low_T=1):
    config = uniform_config()

    seed_all_T_fail = 42
    seed_high_T_pass = 742

    assert mc_step(J, low_T, config, seed_all_T_fail) == None
    assert mc_step(J, low_T, config, seed_high_T_pass) == None


def test_mc_step_uniform_high_T(J=1, high_T=3):
    config = uniform_config()

    seed_all_T_fail = 42
    seed_high_T_pass = 742

    assert mc_step(J, high_T, config, seed_all_T_fail) == None

    expected_pass_config = np.array([[1., 1., 1.],
                                     [1., 1., 1.],
                                     [-1., 1., 1.]])
    expected_pass_energy = -5.0

    queried_pass_config, queried_pass_energy = mc_step(J, high_T, config, seed_high_T_pass)

    assert np.all(queried_pass_config == expected_pass_config)
    assert queried_pass_energy == expected_pass_energy


def test_mc_step_random_low_T(J=1, low_T=1):
    config = random_config()

    seed_all_T_fail = 42
    seed_high_T_pass = 22

    assert mc_step(J, low_T, config, seed_all_T_fail) == None
    assert mc_step(J, low_T, config, seed_high_T_pass) == None


def test_mc_step_random_high_T(J=1, high_T=3):
    config = random_config()

    seed_all_T_fail = 42
    seed_high_T_pass = 22

    assert mc_step(J, high_T, config, seed_all_T_fail) == None

    expected_pass_config = np.array([[-1, 1, -1, -1],
                                     [1, 1, -1, -1],
                                     [-1, 1, -1, -1],
                                     [-1, -1, 1, -1]])
    expected_pass_energy = -2.0

    queried_pass_config, queried_pass_energy = mc_step(J, high_T, config, seed_high_T_pass)

    assert np.all(queried_pass_config == expected_pass_config)
    assert queried_pass_energy == expected_pass_energy


def test_check_convergence():
    converged_x = np.exp(-1 * np.arange(10, 100))
    divergent_x = np.exp(np.arange(10, 100))

    assert check_convergence(converged_x) == True
    assert check_convergence(divergent_x) == None


def test_check_autocorrelation(desired_samples=5):
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    energies = np.load('resources/high_T_4x4_energies_5000.npy')

    assert check_autocorrelation(ensemble, energies, desired_samples, threshold=.01) == 10
    assert check_autocorrelation(ensemble, energies, desired_samples, threshold=.0001) == 106
    assert check_autocorrelation(ensemble, energies, desired_samples, threshold=.000001) == np.inf


def test_run_mcmc_eq(J=1, T=3):
    config = random_config()

    seed_high_T_pass = 22

    expected_eq_config = np.array([[-1, 1, -1, -1],
                                   [1, 1, -1, -1],
                                   [-1, 1, -1, -1],
                                   [-1, -1, 1, -1]])

    queried_eq_config = run_mcmc(J, T, config, seed=seed_high_T_pass)

    assert np.all(queried_eq_config == expected_eq_config)


def test_run_mcmc_ensemble(J=1, T=3, desired_samples=5, min_steps=10):
    config = random_config()

    ensemble, energies = run_mcmc(J, T, config, desired_samples=desired_samples,
                                  min_step_multiplier=.1)

    assert len(ensemble) == desired_samples
    assert len(energies) == desired_samples
