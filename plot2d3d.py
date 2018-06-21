import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


data = load_iris()
y = data.target
X = data.data

#----------------2D plotting-------------------

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)
    
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()


#--------------3D plotting--------------------------

pca = PCA(n_components=3)
reduced_X = pca.fit_transform(X)
    
red_x, red_y, red_z = [], [], []
blue_x, blue_y, blue_z = [], [], []
green_x, green_y, green_z = [], [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
        red_z.append(reduced_X[i][2])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
        blue_z.append(reduced_X[i][2])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
        green_z.append(reduced_X[i][2])

# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(red_x, red_y, red_z, c='r', marker='x')
ax.scatter(blue_x, blue_y, blue_z, c='b', marker='D')
plt.scatter(green_x, green_y, green_z, c='g', marker='o')
plt.show()

