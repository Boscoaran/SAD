#Bosco Aranguren
#Plantilla de algoritmo de prediccion decision tree

import sys
import csv
from unicodedata import name
from wsgiref import headers
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle

hf = []
test = []
trainX = []
trainY = []
testX = []
testY = []
resultados = dict
target_map = {}
clf = object
md = 0
ms = 0
max_fscore=0


def main():
    file=raw_input("Path del archivo .csv: ")
    target=raw_input("Nombre del dato que se quiere calcular: ")
    max_depth=[3,6,9]
    min_samples_leaf=[1, 2]
    get_features(file)
    preparar_datos(file, target)
    with open("resultados_tree.csv", 'w') as csvfile:
        parametros=['max_depth', 'min_samples_leaf', 'class', 'accuracy', 'recall', 'fscore', 'precision']
        writer=csv.DictWriter(csvfile, fieldnames=parametros)
        writer.writeheader
        writer.writerow({'max_depth': "MAX DEPTH", 'min_samples_leaf': "MIN SAMPLES LEAF", 'class': "CLASS", 'accuracy': "ACCURACY", 'recall': "RECALL ", 'fscore': "F-SCORE ", 'precision': "PRECISION"})
    for x in max_depth:
        for y in min_samples_leaf:
            print("Ejecutando algoritmo decision tree con max_depth=" + str(x) + " min_samples_leaf=" + str(y))
            print("\n ################################################## \n")
            decision_tree(x, y)
            print("\n ################################################## \n")
            print("Ejecucion de algoritmo decision tree terminada con max_depth=" + str(x) + " min_samples_leaf=" + str(y) + "\n")
            write_csv(x, y, resultados)
    guardar_modelo=raw_input("Quieres guardar el modelo? S/N: ")
    if guardar_modelo=='S' or guardar_modelo=='s':
        n_modelo = raw_input("Nombre del modelo: ")
        print("Se recomienda usar max_depth = " + str(md) + " y min_samples_leaf = " + str(ms))
        x=int(raw_input("max_depth para el modelo: "))
        y=int(raw_input("min_samples_leaf para el modelo: "))
        print("\n ############################################# \n")
        decision_tree(x,y)
        n_modelo = n_modelo+".sav"
        pickle.dump(clf, open(n_modelo,'wb'))
        print("\n ############################################# \n")
        print("Modelo guardado correctamente empleando Pickle")

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
def preparar_datos(f, t):
    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(f)
    ml_dataset = ml_dataset[hf]

    #CONVERTIR A UNICODE LOS NOMBRES DE COLUMNAS
    #MODIFICAR PARA INTRODUCIR CADA FEATURE EN SU TIPO
    categorical_features = []
    numerical_features = []
    text_features = []
    for feature in ml_dataset:
        if ml_dataset[feature].dtype == object:
            categorical_features.append(feature)
        else:
            numerical_features.append(feature)    
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

    #CALCULAR TARGET MAP
    i=0
    l_unique = pd.unique(ml_dataset[t])
    for unique in l_unique:
        target_map.update({ str(unique): i })
        i += 1   
    #PASAR COLUMNA TARGET(DOUBLE) A __target__(0-1)
    #CAMBIAR DEPENDIENDO DEL TIPO DE FEATURE QUE SE QUIERA CLASIFICAR Y EL NOMBRE DE LA FEATURE
    ml_dataset['__target__'] = ml_dataset[t].map(str).map(target_map)
    del ml_dataset[t]
    hf.remove(t)
    hf.append('__target__')

    #ELIMINAR FILAS EN LAS QUE TARGET ES NULL
    #DF = DF[ ! DF[TARGET] == NULL]     ~ INVIERTE TRUE Y FALSE
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    #DIVIDIR LA MUESTRA EN TRAIN Y TEST. RANDOME_STATE=SEED
    global test
    train,  test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print("Train: ")
    print(train['__target__'].value_counts())
    print("Test: ")
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    impute_when_missing = []

    #SI SE QUIERE ELIMINAR LOS VALORES FALTANTES SE USA DROP_ROWS_WHEN_MISSING, PARA INTRODUCIR VALORES APROX SE USA IMPUT_WHEN_MISSING
    for feature in hf:
        missing = {'feature': feature, 'impute_with': 'MEAN'}
        impute_when_missing.append(missing)

    #ELIMINAR DE TRAIN Y TEST VALORES FALTANTES
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    
    # DEPENDIENDO DE LO QUE SE HAYA ALMACENADO EN IMPUTE_WITH DE LA FEATURE SE CALCULA EL VALOR QUE SE VA A SUSTITUIR
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
        #SUSTITUYE EL VALOR V CALCULADO EN LOS VALORES FALTANTES
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)

    #REESCALADO DE PARAMETROS
    #PARA TODAS LAS FEATURES SE PREPARA EL AVGSTD
    rescale_features = {}
    for feature in hf:
        rescale_features.update({feature : 'AVGSTD'})
 
    #PARA TODAS LAS FEATURES DE TRAIN, SI EL RESCALE METHOD ES MINMAX ______ SINO (AVGSTD) SE CALCULA LA MEDIA Y LA VARIANZA, SI LA VARIANZA ES MUY PEQUENA SE ELIMINA
    #LA FEATURE PORQUE NO PRESENTA CAMBIOS. SI LA VARIANZA ES SIGNIFICATIVA SE LE RESTA A CADA VALOR LA MEDIA Y SE DIVIDE ENTRE LA VARIANZA PARA REESCALARLOS
    '''
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0. and feature_name!='__target__':
            del train[feature_name]
            del test[feature_name]
        elif feature_name!='__target__':
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale  
    '''
    
    global trainX, trainY, testX, testY

    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # RALIZA EN UNDERSAMPLE, LA MAYORIA (MODA) VA A APARECER EL DOBLE DE VECES
    if target_map.values == [0, 1]:
        undersample = RandomUnderSampler(sampling_strategy=0.5)
    else:
        undersample = RandomUnderSampler(sampling_strategy='auto')

    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)

def decision_tree(m_d, m_s_l):
    #USANDO TREE
    global clf
    clf = tree.DecisionTreeClassifier(max_depth=m_d, min_samples_leaf=m_s_l, class_weight="balanced")

    #ENTRENAR ALGORITMO
    clf = clf.fit(trainX, trainY)

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
     for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)
    #CALCULAR RESULTADOS
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
    global resultados
    resultados=classification_report(testY,predictions, output_dict=True)
    resultados.update({'Accuracy': accuracy_score(testY, predictions)})

def write_csv(m_d, m_s_l, resultados):
    with open('resultados_tree.csv', 'a') as csvfile:
        global md
        global ms
        sum_fscore=0
        parametros=['max_depth', 'min_samples_leaf', 'class', 'accuracy', 'recall', 'fscore', 'precision']
        writer=csv.DictWriter(csvfile, fieldnames=parametros)
        writer.writeheader
        for i in resultados.keys()[0:-4]:
            recall=resultados[str(i)]['recall']
            fscore=resultados[str(i)]['f1-score']
            sum_fscore=sum_fscore+fscore
            precision=resultados[str(i)]['precision']
            accuracy=resultados['Accuracy']
            writer.writerow({'max_depth': m_d, 'min_samples_leaf': m_s_l, 'class': i, 'accuracy': accuracy, 'recall': recall, 'fscore': fscore, 'precision': precision})
        av_fscore=sum_fscore/(len(resultados.keys())-4)
        global max_fscore
        if av_fscore>max_fscore:
            max_fscore=av_fscore
            md=m_d
            ms=m_s_l


if __name__=="__main__":
    main()