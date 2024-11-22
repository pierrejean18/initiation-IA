from keras import *
import numpy as np

model = Sequential()
model.add(layers.Dense(units=3,input_shape = [1]))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=1))

entree = np.array([1, 2, 3, 4, 5], dtype=float)
sortie = np.array([2, 4, 6, 8, 10], dtype=float)

model.compile(loss='mean_squared_error',optimizer = 'adam')
model.fit(x=entree, y=sortie,epochs=10000)

while True:
    try:
        x = float(input("Nombre (ou 'exit' pour quitter) : "))
        prediction = model.predict(np.array([x]))[0][0]
        print(f"Prédiction : {prediction}")
    except ValueError:
        print("Programme terminé.")
        break


