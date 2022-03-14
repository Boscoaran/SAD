import pickle

import pandas as pd

file = raw_input("Path del archivo que se quiere clasificar: ")
new_data = pd.read_csv(file)
n_modelo = raw_input("Path del modelo clasificador: ")
clf = pickle.load(open(n_modelo,'rb'))
resultado = clf.predict(new_data)
print(resultado)