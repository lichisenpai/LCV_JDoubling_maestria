import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_json("PLACEHOLDER")

#haciendo limpieza de los datos
dataset_clean = dataset.loc[dataset.loc[:, 'sub_harmonic'] != 0.0] #quitando los valores cuando subarmonico dif a cero 

# define your data
X = dataset_clean[['calculated_J','signal_width', 'sub_harmonic', 'hertz_distance', 'signal_noise_ratio']]
Y = dataset_clean['Jdoubling_error']

#Normalize your data
X_train_stats = X.describe()
X_train_stats = X_train_stats.transpose()
Y_train_stats = Y.describe()
Y_train_stats = Y_train_stats.transpose()
normalizingX = lambda x: (x - X_train_stats['mean'])/ X_train_stats['std'] 
normalizingY = lambda x: (x - Y_train_stats['mean'])/ Y_train_stats['std'] 

normed_Xdata = normalizingX(X)
normed_Ydata = normalizingY(Y)


train_dataset, test_dataset ,train_labels, test_labels = train_test_split(X_pls, normed_Ydata, test_size=0.30, random_state=40)

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

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

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

Lichi.save('./Lichi_NeuralNet1.h5')

#aqui aquí devuelve la perdida, el error absoluto y el error cuadratico 
print("Metricas de desempeño lote de Pruebas:")
loss, mae, mse = Lichi.evaluate(test_dataset, test_labels)

print("Metricas de desempeño lote de Entrenamiento:")
loss, mae, mse = Lichi.evaluate(train_dataset, train_labels)

#haciendo predicciones| viendo la grafica del test vs. predicho 
#desnormalizing = lambda x: (x * train_stats['std'])+train_stats['mean']
#denormed_test_data = desnormalizing(normed_test_data)
print(test_dataset)
test_predictions = Lichi.predict(test_dataset)
print(test_predictions)
plt.scatter(test_labels, test_predictions)
plt.xlabel('Valores Reales/Error [Hz]')
plt.ylabel('Predicciones/ Error [Hz]')
plt.scatter(test_labels, test_labels, color = 'orange')

predecir = np.array([0.5,	-0.444117,	1.012234]).reshape(-1, 3)
prediction = Lichi.predict(predecir).flatten()
print(prediction)

#viendo la distribucion de errores
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("predicción - real (Hz)")
plt.ylabel("Count")