#ML-model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

#load the data
patients = pd.read_csv('preprocessed_data.csv')

#normalization
patients.drop(columns= patients.columns[0], axis=1, inplace=True)
pt_feature = patients.drop('class', axis=1)
x = pt_feature.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
pt_features_norm = pd.DataFrame(x_scaled, columns=pt_feature.columns)

pt_class = [0 if x==0 else 1 for x in patients['class']]

train_features, test_features, train_label, test_label = train_test_split(pt_features_norm, pt_class, test_size=0.3, random_state=0)

#create decided classifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='entropy',random_state=0)
classifier.fit(train_features, train_label)

print(pt_features_norm.iloc[5])
print(classifier.predict([pt_features_norm.iloc[5]]))

#export the ML model
with open(r'prediction_model.pickle', 'wb') as f:
    pickle.dump(classifier, f)
with open(r'scaler.pickle', 'wb') as f:
    pickle.dump(min_max_scaler, f)