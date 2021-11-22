from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap='gray', interpolation="nearest")
    ax.text(0, 7, str(digits.target[i]))
fig.show()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(x_train, y_train)
pre = model.predict(x_test)
print(metrics.classification_report(pre, y_test))
print(metrics.accuracy_score(pre, y_test))
mat = metrics.confusion_matrix(pre, y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.show()
