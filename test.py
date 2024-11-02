import bentoml

# iris_clf_runner = bentoml.sklearn.get().to_runner()
iris_clf = bentoml.sklearn.load_model('iris_clf:latest')
# iris_clf_runner.init_local()
# Make a prediction
print(iris_clf.predict([[5.9, 3.0, 5.1, 1.8]]))

