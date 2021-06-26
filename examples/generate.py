from spin.model import Model
import os
import numpy as np


def gen_samples(j, temp, geo, n_samples):
    lbl = "_".join(
        [
            str(temp)[:4].replace(".", "p"),
            str(geo).replace(",", "x")[1:-1],
            str(n_samples),
        ]
    )
    lbl = lbl.replace(" ", "")
    sample_dir = os.path.join("..", "samples", lbl)
    if os.path.exists(os.path.join(sample_dir, 'model.pkl')):
            return None

    x = Model(J=j, geometry=geo, T=temp, save_path=sample_dir)

    model_file = os.path.join(sample_dir, "model.pkl")
    if not os.path.exists(model_file):
        x.generate_ensemble(n_samples, autocorrelation_threshold=0.5)
        x.generate_rbm()
        x.save_model()
    else:
        return None # x.load_model(model_file)

    # plot_ensemble(x)
    # plot_rbm(x)


tc = 2.0 / np.log(1 + 2 ** 0.5)
temps = np.arange(-1, 1, 0.1)
temps[10] = 0
temps += tc
temps = temps[::-1]

js = [1] # np.random.uniform(1, 2, 3)
print(js)

for j in js:
    for t in temps:
        print(j, t)
        print(gen_samples(j, t, geo=(7, 7), n_samples=5000))
