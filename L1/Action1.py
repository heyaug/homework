import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 载入数据
data = np.load('mnist.npz')

# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data['x_train'], data['y_train'], test_size=0.25, random_state=1)

# 降维
train_x = train_x.reshape(train_x.shape[0], 28 * 28)
test_x = test_x.reshape(test_x.shape[0], 28 * 28)

# 调cart
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)

# 测试
predict = clf.predict(test_x)
print(f"cart准确率:{accuracy_score(predict, test_y):.4f}")

# 86.5
