import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

k_dataset = pd.read_csv('/content/dataset_prisim.csv')

k_dataset.head()
print(k_dataset.shape)

k_dataset["Article"].replace({"YES": 1, "NO": 0}, inplace=True)
k_dataset.head()

k_data=pd.DataFrame({
    'Div Count':k_dataset.iloc[:,0],
    'Span Count':k_dataset.iloc[:,1],
    'Link Count':k_dataset.iloc[:,2],
    'Para Count':k_dataset.iloc[:,3],
    'List Count':k_dataset.iloc[:,4],
    'Image Count':k_dataset.iloc[:,5],
    'Button Count':k_dataset.iloc[:,6],
    'Script Count':k_dataset.iloc[:,7],
    'Article Count':k_dataset.iloc[:,10],
    'Article':k_dataset.iloc[:,12]
    
})

k_X=k_data[['Div Count','Span Count','Link Count', 'Para Count', 'List Count', 'Image Count','Button Count','Script Count','Article Count']]  # Features
k_y=k_data['Article'] 

X_train, X_test, y_train, y_test = train_test_split(k_X, k_y, random_state=1)

count1=0
count0=0

for i in y_test:
  if (i==0):
    count0+=1
  else:
    count1+=1

print(count0)
print(count1)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train.values.ravel())

y_pred = knn.predict(X_test)
print(y_pred)

print(knn.score(X_test, y_test))

v=confusion_matrix(y_test, y_pred)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
     
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(v)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, v[i, j], ha='center', va='center', color='red')
plt.show()

#accurecy calculation at different neighbour value
Tp=1602
Tn=1446

p=count1
n=count0

val=(Tp+Tn)/(p+n)
print('Accurecy.....',val)#value at neighbour=5

# Tp=1601
# Tn=1441

# p=count1
# n=count0

# val=(Tp+Tn)/(p+n)
# print('Accurecy.....',val) #value at neighbour=7
