from sklearn.tree import DecisionTreeRegressor
import numpy as np

tree = DecisionTreeRegressor(min_samples_split=2, max_depth=2)

X = np.random.rand(100, 5)
Y = np.random.randint(0, 2, 100)

tree.fit(X, Y)

t = tree.tree_
x = X[0, :]

path = tree.decision_path(X)
idx = path.indices

print(tree)
