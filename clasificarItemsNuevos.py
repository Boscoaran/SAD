#Bosco Aranguren
#Algoritmo carga de modelo de prediccion

import pandas as pd
import pickle

if __name__ == '__main__':
    d=raw_input("Path de los datos: ")
    m=raw_input("Path del modelo: ")
    
    y_test=pd.DataFrame()
    testX = pd.read_csv(d)

    clf = pickle.load(open(m, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')
    
    
    print(results_test)
    

    
