
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, \
                            recall_score, f1_score, roc_curve, auc
import sklearn
from sklearn.model_selection import GridSearchCV

import joblib
import boto3

import pathlib
from io import StringIO

import argparse
import os

import numpy as np 
import pandas as pd 
from pandas import DataFrame as DF, Series as Srx 


if __name__ == "__main__":
    print('[INFO] Extracting the passed arguments from the command line\n\n')

    parser = argparse.ArgumentParser()
    
    # Hyperparameters sent by the client are passed as command-line arguments to the training script script.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Let's now add parameters for Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training-data-path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--testing-data-path", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--training-file", type=str, default="train.csv")
    parser.add_argument("--testing-file", type=str, default="test.csv")
    # parser.add_argument("--output-data-path", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    known_args, unknown_args = parser.parse_known_args()


    # Let's first of all access the arguments to see their respective values
    n_estimators = known_args.n_estimators
    random_state = known_args.random_state
    model_dir = known_args.model_dir
    train_data_path = known_args.training_data_path # This value is set to 'SM_CHANNEL_TRAIN' as an env var after the "mode.fit({'train': d1, 'test': d2})"
    test_data_path = known_args.testing_data_path # This value is set to 'SM_CHANNEL_TEST' as an env var after the "mode.fit({'train': d1, 'test': d2})"
    train_file = known_args.training_file
    test_file = known_args.testing_file

    print(f"Number of estimators: {n_estimators}")
    print(f"Random state: {random_state}")
    print(f"Model directory: {model_dir}")
    print(f"Training data path: {train_data_path}")
    print(f"Testing data path: {test_data_path}")
    print(f"Training file: {train_file}")
    print(f"Testing file: {test_file}")

    # print("SkLearn Version: ", sklearn.__version__)
    # print("Joblib Version: ", joblib.__version__)

    print('\n\n')
    print("[INFO] Reading data from the S3 channels")
    print()

    training_data_df: DF = pd.read_csv(os.path.join(known_args.training_data_path, known_args.training_file))
    testing_data_df: DF = pd.read_csv(os.path.join(known_args.testing_data_path, known_args.testing_file))
    
    features = list(training_data_df.columns)
    # label = features.pop(-1)

    print("Building training and testing datasets")
    print()

    # Train independent features dataframes
    X_train_df: DF = training_data_df[features]
    # Train dependent feature
    y_train = training_data_df['price_range']



    # Test independent features dataframes
    X_test_df: DF = testing_data_df[features]  
    # Test dependent feature
    y_test = testing_data_df['price_range']

    # print('Column order: ')
    # print(features)
    # print()
    
    # print("Label column is: ", label)
    # print()
    
    print("\nTraing and testing dataframes shapes: ")
    print()
    
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(f'X_train shape: \t {X_train_df.shape}\n')
    print(f'y_train shape: \t {y_train.shape}\n')
    print()
    
    print("---- SHAPE OF TESTING DATA (15%) ----")
    print(f'X_test shape: \t {X_test_df.shape}\n')
    print(f'y_test shape: \t {y_test.shape}\n')

    print('\n\n')
    
  
    print("Training RandomForest Model.....")
    print()

    # Define the parameter grid
    param_grid = {
        # 'n_estimators': [50, 100, 200], # Let me remove this hyper parameter since it will be passed as a command line argument. 
        # I could choose to pass the whole list [50, 100, 200] as a command line argument by using the argparse's 'nargs' 
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    

    # Let's first of all initialize the model
    model =  RandomForestClassifier(n_estimators=known_args.n_estimators, random_state=known_args.random_state, verbose = 3,n_jobs=-1)
    
    
    # And then, initialize Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)


    # And then let's fit Grid Search
    # model.fit(X_train, y_train)
    grid_search.fit(X_train_df.values, y_train.values)
    
    print(f'\nGrid Search best parameters: \n {grid_search.best_estimator_}')



    # Let's Get the best model
    best_model = grid_search.best_estimator_

    print()
    
    # Let's specify where we are going to dump our model
    # model_path = os.path.join(known_args.model_dir, "model.joblib")
    # joblib.dump(model,model_path)


    # Let's now, save the best model
    model_path = os.path.join(known_args.model_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)

    
    print(f'The best model is  persisted at:\t {model_path}')


    print()

    
    test_y_pred = best_model.predict(X_test_df)
    test_accuracy = accuracy_score(y_test,test_y_pred)


    test_classification_report = classification_report(y_test,test_y_pred)

    print('\n\n')


    print("---- METRICS RESULTS FOR TESTING DATA ----")

    print()
    
    print(f'Total number of testing data points is \t: {X_test_df.shape[0]}\n\n')

    print(f'Testing accuracy is \t: {test_accuracy}')

    print(f'Testing classification report is \n: {test_classification_report}')
    
    print('\n\n')
    
    print('END !!!')
