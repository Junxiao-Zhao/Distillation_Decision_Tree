from sklearn.model_selection import train_test_split
from induction_ddt import DDT
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import boston_housing

#train the Neural Network
def fit_model(train_data, test_data, train_targets):

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std

    model=keras.models.Sequential()
    model.add(keras.layers.InputLayer([train_data.shape[1], ]))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

    h = model.fit(train_data, train_targets,
            batch_size=1,
            epochs=100,
            verbose=1,
            validation_data=(test_data, test_targets))

    model.save("boston_housing.h5")

model = keras.models.load_model('boston_housing.h5')

#Prediction method
def model_predict(df:pd.DataFrame, mean:float, std:float):
    train_data = df.values.copy()
    train_data -= mean
    train_data /= std

    results = model.predict(train_data)
    return results

if __name__ == "__main__":
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    data = pd.DataFrame(test_data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', "LSTAT"])
    data.insert(data.shape[1], "label", test_targets)

    des = data.describe().loc[['min', 'max'],:]
    attr_info = des.to_dict()

    label_info = attr_info["label"]
    del attr_info['label']

    #generate DDT
    ddt = DDT(model_predict, attr_info, label_info, train_data)
    """ root = ddt.fit(stopping_criteria=2, pca=False, num_of_iter=10, min_sample_each=10, stability=(0.2,0.7), 
                            lgb_para={"learning_rate":0.09, "max_depth":-5, "random_state": 42}, 
                            train_test_split_para={'test_size': 0.33, 'random_state': 42}, 
                            fit_para={'verbose': 20, 'eval_metric': 'logloss'})
    
    ddt.save("root.dat") """

    data.drop('label', inplace=True, axis=1)

    root = ddt.load("root.dat")

    pred = ddt.predict(root, data)