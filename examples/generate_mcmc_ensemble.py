from spin.model import Model

ensemble_size = 250

model = Model(
    T=3.16,
    geometry=(40, 40),
    save_path="../samples/3p16_40x40_{}".format(ensemble_size),
)
model.generate_ensemble(ensemble_size, autocorrelation_threshold=0.5)
model.save_model("model_0.pkl")
