import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

max_accuracy=0
for learning_rate in [0.0001,0.001,0.1,0.3,0.5,0.75,1.0,1.5,2.0]:
    model = LogisticRegression(alpha =learning_rate)
    model.fit(X_train_norm,y_train)
    predictions = model.predict(X_test_norm)
    accuracy = np.sum(y_test==predictions)/len(y_test)
    print(f"Accuracy of the model with learning rate {learning_rate} is {accuracy}")
    if accuracy>max_accuracy:
        max_accuracy = max(max_accuracy,accuracy)
        best_lr = learning_rate
print(f"The model achieved maximum accuracy at learning rate {best_lr}")



