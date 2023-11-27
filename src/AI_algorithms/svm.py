"""An SVM to predict whether Jdoubling's output is reliable using signal_to_noise_ratio and the Jdoubling output"""
import pickle
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

CURRENT_DIR = path.dirname(__file__)
INPUT_DB = path.join(CURRENT_DIR, "../../assets/trainning-databases/old_noise-no_flat.json")
SVM_MODEL_FILENAME = path.join(CURRENT_DIR, "udata/svm-model.pkl")


def save_svm_model(model):
    serialized_model = pickle.dumps(model)
    with open(SVM_MODEL_FILENAME, mode="wb") as file:
        file.write(serialized_model)

def load_svm_model():
    with open(SVM_MODEL_FILENAME, mode="rb") as file:
        serialized = file.read()
        return pickle.loads(serialized)

class JdoublingReliabilityPredictor:
    def __init__(self):
        self.svm = load_svm_model()
    
    def predict(self, signal_noise_ratio: float, calculated_j: float) -> float:
        row = {
            "signal_noise_ratio": [signal_noise_ratio],
            "calculated_J": [calculated_j]
        }
        row = pd.DataFrame(data=row)

        vals = self.svm.predict(row)
        return vals[0]

if __name__ == "__main__":
    # Data preprocessing ----------------------
    dataset = pd.read_json(INPUT_DB)
    
    def max_aceptado (x):
       return x * 0.95
    
    def min_aceptado (x):
       return x * 1.05 
    
    #haciendo limpieza de los datos
    dataset_clean = dataset.loc[dataset.loc[:, 'sub_harmonic'] != 0.0] #quitando los valores cuando subarmonico dif a cero 
    dataset_clean['Max_acept'] = dataset_clean['theoretical_J'].apply(max_aceptado)
    dataset_clean['Min_acept'] = dataset_clean['theoretical_J'].apply(min_aceptado)
    
    print("Dataset limpio\n", dataset_clean.tail())
    
    # define your data
    X = dataset_clean[['signal_noise_ratio', 'calculated_J']]
    #Y = dataset_clean['Cluster'] = [0 if s >= 0.15 or a <= 100 else 1 for s, a in zip(X['Error'], X['S/n'])]
    condiciones = [(dataset_clean.calculated_J >= dataset_clean.Max_acept) & (dataset_clean.calculated_J <= dataset_clean.Min_acept), 
                   (dataset_clean.calculated_J < dataset_clean.Max_acept) & (dataset_clean.calculated_J > dataset_clean.Min_acept)]
    
    elecciones = np.array((0, 1), dtype="int8")
    Y = dataset_clean['Cluster'] = np.select(condiciones, elecciones, -1)
    #Grafica
    colores = ["#00cc44", "#ff0000"]   # Green & Red 
    plt.scatter(X['signal_noise_ratio'], X['calculated_J'], c=np.take(colores, Y), s=10)
    plt.xlabel("S/n ratio")
    plt.ylabel("Jdet (Hz)")
    
    # Radial kernel --------------------
    
    #sepparating data
    train_dataset, test_dataset ,train_labels, test_labels = train_test_split(X, Y, test_size=0.30, random_state=40)
    
    # Grid de hiperparámetros
    param_grid = {'C': np.logspace(-5, 7, 20)}
    
    # Búsqueda por validación cruzada
    grid = GridSearchCV(
            estimator  = SVC(kernel= "rbf", gamma='scale'),
            param_grid = param_grid,
            scoring    = 'accuracy',
            n_jobs     = -1,
            cv         = 3, 
            verbose    = 0,
            return_train_score = True
          )
    
    _ = grid.fit(train_dataset, train_labels)
    
    # Mejores hiperparámetros por validación cruzada
    print("Mejores hiperparámetros encontrados (cv)")
    print(grid.best_params_, ":", grid.best_score_, grid.scoring)
    
    modelo = grid.best_estimator_
    
    # Representación gráfica de los límites de clasificación
    
    # Grid de valores
    x = np.linspace(np.min(train_dataset['signal_noise_ratio']), np.max(train_dataset['signal_noise_ratio']), 50)
    y = np.linspace(np.min(train_dataset['calculated_J']), np.max(train_dataset['calculated_J']), 50)
    Y, X = np.meshgrid(y, x)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Predicción valores grid
    pred_grid = modelo.predict(grid)
    
    fig, ax = plt.subplots(figsize=(12,8))
    #ax.scatter(grid[:,0], grid[:,1], c=pred_grid, alpha = 0.2)
    ax.scatter(train_dataset['signal_noise_ratio'], train_dataset['calculated_J'], c=train_labels, alpha = 1, s =9)
    
    # Vectores soporte
    ax.scatter(
        modelo.support_vectors_[:, 0],
        modelo.support_vectors_[:, 1],
        s=5, linewidth=1,
        facecolors='none', edgecolors='black'
    )
    
    # Hiperplano de separación
    ax.contour(
        X, Y,modelo.decision_function(grid).reshape(X.shape),
        colors='k',
        levels=[0],
        alpha=0.5,
        linestyles='-'
    )
    ax.set_title("Resultados clasificación SVM radial")
    
    # Accuracy de test del modelo 
    predicciones = modelo.predict(test_dataset)
    accuracy = accuracy_score(
                y_true    = test_labels,
                y_pred    = predicciones,
                normalize = True
               )
    print("")
    print(f"El accuracy de test es: {100*accuracy}%")

    # Matriz de confusión de las predicciones de test
    confusion_matrix = pd.crosstab(
        test_labels.ravel(),
        predicciones,
        rownames=['Real'],
        colnames=['Predicción']
    )
    print(confusion_matrix)

    save_svm_model(modelo)
