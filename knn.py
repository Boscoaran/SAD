#Bosco Aranguren
#Plantilla de algoritmo de prediccion knn

import getopt
import sys
import os
import csv
from unicodedata import name
from wsgiref import headers
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

oFile="output.out"
hf = []

#GESTIONA LOS BUCLES DE K Y P
def main():
    file=raw_input("Path del archivo .csv: ")
    get_features(file)
    for k in range(1, 2, 2):
        for p in range(1, 2, 2):
            knn(file,k,p)
            print("Ejecucion de algoritmo knn terminada con k=" + str(k) + " p=" + str(p))


#OBTIENE EL NOMBRE DE LAS FEATURES Y LO ALMACENA EN LA VARIBALE GLOBAL hf (headers features)
def get_features(f):
    with open(f, 'r') as file:
        d_reader = csv.DictReader(file)
        global hf
        hf = d_reader.fieldnames

#COERCE A UNICODE
def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)

#ALGORITMO KNN QUE RECIBE: f: DIRECCION DEL FICHERO, k, p, hf: headers features
def knn(f, k, p):
    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(f)
    ml_dataset = ml_dataset[hf]

    #CONVERTIR A UNICODE LOS NOMBRES DE COLUMNAS
    #MODIFICAR PARA INTRODUCIR CADA FEATURE EN SU TIPO
    categorical_features = []
    numerical_features = hf
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    #COMPROBAR SI EL DATO ES UNA FECHA-HORA O NO PARA GUARDARLO COMO DOUBLE
    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]'): #or (hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]'))
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    #PASAR COLUMNA TARGET(DOUBLE) A __target__(0-1)
    #CAMBIAR DEPENDIENDO DEL TIPO DE FEATURE QUE SE QUIERA CLASIFICAR Y EL NOMBRE DE LA FEATURE
    target_map = {'0.0': 0, '1.0': 1}
    ml_dataset['__target__'] = ml_dataset['TARGET'].map(str).map(target_map)
    del ml_dataset['TARGET']

    #ELIMINAR FILAS EN LAS QUE TARGET ES NULL
    #DF = DF[ ! DF[TARGET] == NULL]     ~ INVIERTE TRUE Y FALSE
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    '''

    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    impute_when_missing = []
    for feature in hf:
        missing = {'feature': feature, 'impute_with': 'MEAN'}
        impute_when_missing.append(missing)

    # Explica lo que se hace en este paso
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Explica lo que se hace en este paso
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    #REESCALADO DE PARAMETROS
    rescale_features = {}
    for feature in hf:
        rescale_features.update({hf : 'AVGSTD'})
        
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
    train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
    test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    drop_rows_when_missing = []
    impute_when_missing = []
    for feature in hf:
        missing = {'feature': feature, 'impute_with': 'MEAN'}
        impute_when_missing.append(missing)

        # Features for which we drop rows with missing values"
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Explica lo que se hace en este paso
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    rescale_features = {}
    for feature in hf:
        rescale_features.update({hf : 'AVGSTD'})

    # Explica lo que se hace en este paso
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # Explica lo que se hace en este paso
    undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # Explica lo que se hace en este paso
    clf = KNeighborsClassifier(n_neighbors=5,
                              weights='uniform',
                              algorithm='auto',
                              leaf_size=30,
                              p=2)

        # Explica lo que se hace en este paso

    clf.class_weight = "balanced"
        # Explica lo que se hace en este paso

    clf.fit(trainX, trainY)


    # Build up our result dataset

    # The model is now trained, we can apply it to our test set:

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)

    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
     for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

    # Build scored dataset
    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test['__target__'], how='left')
    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

    i=0
    for real,pred in zip(testY,predictions):
        print(real,pred)
        i+=1
        if i>5:
            break

    print(f1_score(testY, predictions, average=None))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions, labels=[1,0]))
    print("FIN")
    '''

if __name__=="__main__":
    main()