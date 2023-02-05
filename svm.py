import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

article_dataset = pd.read_csv('/content/drive/MyDrive/content/articledataset.csv')

article_dataset.head()
#number of rows ad columns in the dataset
article_dataset.shape

#converting YES to 1 and NO to 0 in the articles column
article_dataset["Article"].replace({"YES": 1, "NO": 0}, inplace=True)
#getting the statistical details about the data
article_dataset.describe()
#1 is for articles and 0 represents non-articles
article_dataset['Article'].value_counts()
#to compare and find differentiating factors
article_dataset.groupby('Article').mean()
#Separating the data and labels
X = article_dataset.drop(columns = ['Article','Link'], axis=1)
Y = article_dataset['Article']
print(X)
print(Y)

#Standardization

scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = article_dataset['Article']

print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (209,192,102,0,77,9,1,34,1,0,0,9767)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
#print(prediction)

if (prediction[0] == 0):
  print('Not an article')
else:
  print('Article')
