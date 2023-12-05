import pickle
from os import path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_normalization import normalize_training_data, Normalizer

CURRENT_DIR = path.dirname(__file__)
INPUT_DB = path.join(CURRENT_DIR, "../../assets/training-databases/old_noise-no_flat.json")
PLS_MODEL_FILENAME = path.join(CURRENT_DIR, "udata/nn-dimensional-projection.pkl")
NEURAL_NET_MODEL_FILENAME = path.join(CURRENT_DIR, "udata/lichi-net.keras")


def save_pls_model(model):
    serialized_model = pickle.dumps(model)
    with open(PLS_MODEL_FILENAME, mode="wb") as file:
        file.write(serialized_model)

def load_pls_model():
    with open(PLS_MODEL_FILENAME, mode="rb") as file:
        serialized = file.read()
        return pickle.loads(serialized)

class JdoublingErrorPredictor:
    def __init__(self):
        # load both models into memory
        self.neural_net_fwd_pass = tf.saved_model.load(NEURAL_NET_MODEL_FILENAME)
        self.pls_model = load_pls_model()
        self.normalizer = Normalizer()
   
    def predict(self, calculated_j:float, signal_width: float, sub_harmonic: float, hertz_distance: float, signal_noise_ratio: float) -> float:
        row = [calculated_j, signal_width, sub_harmonic, hertz_distance, signal_noise_ratio]

        row = {
            "calculated_J":[calculated_j],
            "signal_width":[signal_width],
            "sub_harmonic":[sub_harmonic],
            "hertz_distance":[hertz_distance],
            "signal_noise_ratio":[signal_noise_ratio]
        }
        row = pd.DataFrame(data=row)

        normalized_input = self.normalizer.normalize_row(row)
        reduced_dimensions = self.pls_model.transform(normalized_input)
        result = self.neural_net_fwd_pass.serve(reduced_dimensions).numpy()[0,0]
        result = self.normalizer.denormalize_row({"Jdoubling_error": result})
        return result["Jdoubling_error"]

if __name__ == "__main__":
    dataset = normalize_training_data(INPUT_DB)
    normed_Xdata = dataset[['calculated_J','signal_width', 'sub_harmonic', 'hertz_distance', 'signal_noise_ratio']]
    normed_Ydata = dataset['Jdoubling_error']

    ##PLS reduction to three columns
    pls = PLSRegression(n_components=3)
    pls.fit(normed_Xdata, normed_Ydata)
    save_pls_model(pls)

    X_pls = pls.transform(normed_Xdata)
    column_names = ["Column_1", "Column_2", "Column_3"]
    X_transformed = pd.DataFrame(X_pls, columns=column_names)
    
    #Grafica para ver como se ven las componentes
    plt.scatter(X_transformed['Column_1'], normed_Ydata, label= "Componente 1")
    plt.scatter(X_transformed['Column_2'], normed_Ydata, label= "Componente 2")
    plt.scatter(X_transformed['Column_3'], normed_Ydata, label= "Componente 3")
    plt.legend()
    plt.xlabel("Datos proyectados en la componente $n$")
    plt.ylabel("Error experimental en J Doubling $(Hz)$")
    
    train_dataset, test_dataset ,train_labels, test_labels = train_test_split(X_pls, normed_Ydata, test_size=0.30, random_state=40)

    ## Building  model----------
    def NeuralNetwork ():
        lichi = keras.Sequential([
            layers.Dense(6, activation='sigmoid', input_shape=[3]),
            layers.Dense(6, activation='sigmoid'),
            layers.Dense(6, activation='relu'),
            layers.Dense(1)
            ])
    
        optimizer = tf.keras.optimizers.Adam()
    
        lichi.compile(loss='mse',optimizer=optimizer, metrics=['mae', 'mse'])
        return lichi
    
    Lichi = NeuralNetwork()
    Lichi.summary()
    
    ## Training----------
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')
    
    EPOCHS = 500
    
    def plot_history(history):
      hist = pd.DataFrame(history.history)
      hist['epoch'] = history.epoch
    
      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Error Absoluto Medio [Hz]')
      plt.plot(hist['epoch'], hist['mae'],
               label='Train Error')
      plt.plot(hist['epoch'], hist['val_mae'],
               label = 'Val Error')
      plt.legend()
    
      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Error Cuadrático Medio [$Hz^2$]')
      plt.plot(hist['epoch'], hist['mse'],
               label='Train Error')
      plt.plot(hist['epoch'], hist['val_mse'],
               label = 'Val Error')
      plt.legend()
      plt.show()
    
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = Lichi.fit(train_dataset, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    plot_history(history)
    Lichi.export(NEURAL_NET_MODEL_FILENAME)
    
    ## Training results----------
    #aqui aquí devuelve la perdida, el error absoluto y el error cuadratico 
    print("Metricas de desempeño lote de Pruebas:")
    loss, mae, mse = Lichi.evaluate(test_dataset, test_labels)
    
    print("Metricas de desempeño lote de Entrenamiento:")
    loss, mae, mse = Lichi.evaluate(train_dataset, train_labels)
    
    print(test_dataset)
    test_predictions = Lichi.predict(test_dataset)
    print(test_predictions)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('Valores Reales/Error [Hz]')
    plt.ylabel('Predicciones/ Error [Hz]')
    plt.scatter(test_labels, test_labels, color = 'orange')
