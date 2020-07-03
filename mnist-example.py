import numpy as np
from tensorflow.python.keras import datasets
from ELM import ELM

(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

x = ELM(500)
x.fit(X_train, Y_train)
y = x.predict(X_test)

correct = 0
total = y.shape[0]
for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(Y_test[i])
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))
