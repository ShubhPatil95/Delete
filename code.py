import yaml
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score, precision_score
from sklearn.linear_model import LogisticRegression
import argparse

def Create_Model(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)  
    # Label Encoded Dataset
    # Iris-setosa = 0
    # Iris-versicolor = 1
    # Iris-virginica = 2
    ## import dataset
    data_path = config["data_source"]["local_data_path"]
    df = pd.read_csv(data_path)
    
    ## Separate dependent and independent features
    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    
    ## Split the into training and testing
    test_size = config["base"]["test_size"]
    seed = config["base"]["random_state"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
    
    ## Create Logistic regression model
    fit_int = config["model"]["logistic_regression"]["fit_intercept"]
    multi_class = config["model"]["logistic_regression"]["multi_class"]
    log_reg = LogisticRegression(fit_intercept=fit_int, penalty='l2', 
                                 multi_class=multi_class,
                                 random_state = seed)
    log_reg.fit(x_train,y_train)
    y_pred = log_reg.predict(x_test)
    
    ## Metrics 
    average = config["model"]["logistic_regression"]["average"]
    accuracy = accuracy_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test,average=average)
    precision = precision_score(y_pred, y_test,average=average)
    
    print("Accuracy==>",accuracy)
    print("Recall==>",recall)
    print("Precision==>",precision)
    
    scores_file=config["reports"]["scores"]
    params_file=config["reports"]["params"]
    
    with open(scores_file,"w") as score:
        scores={
            "Accuracy":accuracy,
            "Recall": recall,
            "Precision": precision
            }
        json.dump(scores,score,indent=4)
        
    with open(params_file, "w") as param:
        params = {
            "test_size": test_size,
            "random_state": seed,
            "fit_intercept": fit_int,
            "multi_class": multi_class        
        }
        json.dump(params, param, indent=4)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/home/shubham/DVC/DVC_Metrics_Tracking/params.yaml")
    parsed_args = args.parse_args()
    Create_Model(config_path=parsed_args.config)

    

