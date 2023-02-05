import pandas as pd
import numpy as np

r_dataset = pd.read_csv('/content/dataset_prisim.csv')

r_dataset.head()
r_dataset.shape

r_dataset["Article"].replace({"YES": 1, "NO": 0}, inplace=True)
r_X = r_dataset.iloc[:, :11].values
r_y = r_dataset.iloc[:, 12].values

print(r_X,r_y)
r_dataset.iloc[:,:13]

r_data=pd.DataFrame({
    'Div Count':r_dataset.iloc[:,0],
    'Span Count':r_dataset.iloc[:,1],
    'Link Count':r_dataset.iloc[:,2],
    'Para Count':r_dataset.iloc[:,3],
    'List Count':r_dataset.iloc[:,4],
    'Image Count':r_dataset.iloc[:,5],
    'Button Count':r_dataset.iloc[:,6],
    'Script Count':r_dataset.iloc[:,7],
    'Article Count':r_dataset.iloc[:,10],
    'Article':r_dataset.iloc[:,12]
    
})

from sklearn.model_selection import train_test_split

r_X=r_data[['Div Count','Span Count','Link Count', 'Para Count', 'List Count', 'Image Count','Button Count','Script Count','Article Count']]  # Features
r_y=r_data['Article']  # Labels

# Split dataset into training set and test set
r_X_train, r_X_test, r_y_train, r_y_test = train_test_split(r_X, r_y, test_size=0.25) # 70% training and 30% test

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
Article_classifier=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
Article_classifier.fit(r_X_train,r_y_train)

feature_imp = pd.Series(Article_classifier.feature_importances_,index=['Div Count','Span Count', 'Link Count', 'Para Count', 'List Count', 'Image Count',
       'Button Count', 'Script Count', 'Article Count']).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
Article_classifier=RandomForestClassifier(bootstrap = True,
                               max_features = 'sqrt')

#Train the model using the training sets y_pred=clf.predict(X_test)
Article_classifier.fit(r_X_train,r_y_train)

r_y_pred=Article_classifier.predict(r_X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(r_y_test, r_y_pred))

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(r_y_test, r_y_pred, average='macro')

precision_recall_fscore_support(r_y_test, r_y_pred, average='micro')
precision_recall_fscore_support(r_y_test,r_y_pred, average='weighted')

