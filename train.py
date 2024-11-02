import bentoml
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

#train the model
clf = svm.SVC(gamma = 'scale')
clf.fit(X, y)

# Save the model to the BentoML local modal store
saved_model = bentoml.sklearn.save_model('iris_clf', clf) 
print(f'model saved : {saved_model}')
# model was stored at this location C:\Users\user\bentoml\models\iris_clf\qakxromzjw7jt2gy