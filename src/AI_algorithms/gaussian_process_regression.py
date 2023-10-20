from os import path
import pickle

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression, PLSSVD
import gpflow #probablemente tengas que instalarlo manualmente con !pip install gpflow y volver a reiniciar el entorno 
from gpflow.utilities import print_summary
import tensorflow as tf

from data_normalization import normalize_training_data, Normalizer

CURRENT_DIR = path.dirname(__file__)
INPUT_DB = path.join(CURRENT_DIR, "../../assets/old_noise-no_flat.json")
PLS_MODEL_FILENAME = path.join(CURRENT_DIR, "udata/gpr-dimensional-projection.pkl")
GPR_MODEL_DIR = path.join(CURRENT_DIR, "udata/lichi-gpr/")

def save_pls_model(model):
    serialized_model = pickle.dumps(model)
    with open(PLS_MODEL_FILENAME, mode="wb") as file:
        file.write(serialized_model)

def load_pls_model():
    with open(PLS_MODEL_FILENAME, mode="rb") as file:
        serialized = file.read()
        return pickle.loads(serialized)

class JdoublingResultPredictor:
    def __init__(self):
        self.pls_model = load_pls_model()
        self.gpr_model = tf.saved_model.load(GPR_MODEL_DIR)
        self.normalizer = Normalizer()

    def predict(self, signal_width: float, subharmonic: float, hertz_distance: float, signal_noise_ratio: float) -> float:
        row = [signal_width, subharmonic, hertz_distance, signal_noise_ratio]

        row = {
            "signal_width": [signal_width],
            "sub_harmonic": [subharmonic],
            "hertz_distance": [hertz_distance],
            "signal_noise_ratio": [signal_noise_ratio]
        }
        row = pd.DataFrame(data=row)

        normalized_input = self.normalizer.normalize_row(row)
        reduced_dimensions = self.pls_model.transform(normalized_input)
        prediction = self.gpr_model.predict_f_compiled(reduced_dimensions)
        (predicted_J, variance) = tf.get_static_value(prediction)
        #extract from tensor    
        (predicted_J, variance) = (predicted_J.numpy()[0,0], variance.numpy()[0,0])
        predicted_J = self.normalizer.denormalize_row({"calculated_J": predicted_J})
        predicted_J = predicted_J["calculated_J"]
        return (predicted_J, variance)
        
if __name__ == "__main__":
    dataset = normalize_training_data(INPUT_DB)
    normed_Xdata = dataset[['signal_width', 'sub_harmonic', 'hertz_distance', 'signal_noise_ratio']]
    normed_Ydata = dataset['calculated_J']

    # PLS
    pls = PLSRegression(n_components=3)
    pls.fit(normed_Xdata, normed_Ydata)
    save_pls_model(pls)

    X_pls = pls.transform(normed_Xdata)
    Y_pls = pls.predict(normed_Xdata)

    column_names = ["Column_1", "Column_2", "Column_3"]
    X_transformed = pd.DataFrame(X_pls, columns=column_names)
    Y_transformed = pd.DataFrame(Y_pls, columns=["Y_predicted"])

    # Muestreo aleatorio simple
    # Debemos obtener una muestra representativa de los datos por que la compeljidad
    # de la regresión gaussiana es n**3
    # Para esto usamos la formula de población finita:
    # n= (Z^2 * N * p * q)/(e^2 * (N-1) + Z^2 * p * q
    # Z: 1.96 <- which means 95% confidence
    # e: 0.02 <- max error
    def get_sample_size(data_size: int) -> int:
        z = 1.96
        e = 0.02
        res = (z**2 * data_size * 0.5 * 0.5) / (e**2 * (data_size - 1) + 0.5 * 0.5)
        return int(res)

    sample_size = get_sample_size(len(X_transformed))

    X_sampledd =X_transformed.sample(n=sample_size, random_state= 1)
    X_sampled = X_sampledd.to_numpy()
    Y_sampledd = normed_Ydata.sample(n=sample_size, random_state= 1)
    Y_sampled= Y_sampledd.to_numpy().reshape(-1,1)

    # Gaussian Process Regression
    #Elijiendo kernel
    k1 = gpflow.kernels.Polynomial(active_dims=[0])
    k2 = gpflow.kernels.Polynomial(active_dims=[1])
    k3 = gpflow.kernels.Polynomial(active_dims=[2])
    k = k1 + k2 + k3

    #Construyendo el modelo
    lichi_gpr = gpflow.models.GPR(data=(X_sampled, Y_sampled), kernel=k, mean_function=None)

    #Optimizadores 
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(lichi_gpr.training_loss, lichi_gpr.trainable_variables, options=dict(maxiter=100))
    print("Summary model after optimization: \n")
    print_summary(lichi_gpr)

    # Saving model
    lichi_gpr.predict_f_compiled = tf.function(
        lichi_gpr.predict_f, input_signature=[tf.TensorSpec(shape=[1, 3], dtype=tf.float64)]
    )
    tf.saved_model.save(lichi_gpr, GPR_MODEL_DIR)

    ## prediction
    #haciendo una prediccion 
    #entrada = np.array([0.75, -0.34, -1.15]).reshape(-1,3)
    #prediciendo = m.predict_f(entrada)
    #media, varianzaa = m.predict_f(entrada) 
    #
    #print("Media: ",media, "; Varianza K=2 :", varianzaa)
