import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruit_data = pd.read_table('fruit_data_with_colors.txt')
fruit_data.head

#creating fruit label pair
fruit_name_label = dict(zip(fruit_data.fruit_label.unique(), fruit_data.fruit_name.unique()))
print(fruit_name_label)

#creating X and y for plotting
X = fruit_data[['mass', 'width', 'height', 'color_score']]
Y = fruit_data['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)

## plotting a 3D scatter plot
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
#ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
#ax.set_xlabel('width')
#ax.set_ylabel('height')
#ax.set_zlabel('color_score')
#plt.show()

# X and y for training
X = fruit_data[['mass', 'width', 'height']]
Y = fruit_data['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#Creating the classifier
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

#using classifier to classify new fruit
new_fruit = [[170, 5.3, 5.5], [290, 6.1, 6.5]]
prediction = knn.predict([new_fruit[0]])
print(fruit_name_label[prediction[0]])

prediction = knn.predict([new_fruit[1]])
print(fruit_name_label[prediction[0]])

## effect of K on accuracy
#k_range = range(1, 20)
#scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors = k)
#    knn.fit(X_train, y_train)
#    scores.append(knn.score(X_test, y_test))
#plt.figure()
#plt.xlabel('k')
#plt.ylabel('accuracy')
#plt.scatter(k_range, scores)
#plt.xticks([0,5,10,15,20])
#plt.show()

h = 0.1
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

XX, YY = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.array([XX.ravel(), YY.ravel()]).T

print(Z.shape)





##Z = Z.reshape(XX.shape)
#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#plt.figure()
#plt.pcolormesh(XX, YY, Z, cmap=cmap_light)
#
#plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
#plt.xlim(XX.min(), XX.max())
#plt.ylim(YY.min(), YY.max())
##plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))