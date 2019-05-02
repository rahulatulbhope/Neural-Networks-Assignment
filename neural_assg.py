import numpy as np
import pandas as pd

dataset = pd.read_csv("train_x_labels.csv")

X = dataset.iloc[:, 0:12].values
y = dataset.iloc[:, 12].values


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

mlp.fit(X,y)

MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

#predictions = mlp.predict(X_test)
X_test_new = pd.read_csv("test_x_labels.csv")
#print(X_test_new.shape)
X_test_new_1 = X_test_new.iloc[:, 0:12].values
y_new = X_test_new.iloc[:, 12].values
#print(X_test_new_1.shape)
#print(y_new.shape)
predictions_1 = mlp.predict(X_test_new_1)



from sklearn.metrics import accuracy_score,confusion_matrix
#print(confusion_matrix(y_test,predictions))

print(confusion_matrix(y_new,predictions_1))
print(accuracy_score(y_new,predictions_1)*100)