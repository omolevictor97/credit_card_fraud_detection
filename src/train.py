from sklearn.ensemble import RandomForestClassifier 
from src.constants import selected_list
from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
import pickle

def model_train(X_train, y_train, train="X_train"):
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2, max_features=10, min_samples_leaf=3, min_samples_split=3)
    if train != "X_train":
        model.fit(X, y)
        return model, X, y
    else:
        X_train_selected = X_train[selected_list]
        X_test_selected = X_test[selected_list]
        model.fit(X_train_selected, y_train)

        return model, X_train_selected, X_test_selected
    

if __name__ == "__main__":
    # Call the class data ingestion
    ingested_data = DataIngestion(data_path="data/application_data.csv")
    data = ingested_data.ingest_data()
    transform_data = DataTransformation(data)

    X, y, X_train, X_test, y_train, y_test = transform_data.transform()
    model, X_train, X_test = model_train(X_train=X_train, y_train=y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("Test score is", test_acc)
