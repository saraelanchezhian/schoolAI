import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from classifier import *
from report import *

def daily_predict(inputs, daily_inputs):

    """
        Initializing threat level classifier model to train data.
    """
    model = ThreatLevelClassifier(inputs)
    model.train()
    
    """
        Invoke model to predict and classfiy output for test data.
    """
    y_pred = model.predict(daily_inputs)
    y_test = daily_inputs.iloc[:,32]

    """
        Calculate accuracy using these methods:
        1. accuracy_score
        2. Precision and recall using confusion_matrix
        
    """
    accuracy = accuracy_score(y_test, y_pred)
    accuracy2 = confusion_matrix(y_test, y_pred)
    prec_recall = precision_recall_fscore_support(y_test, y_pred, average='macro')
    disp = ConfusionMatrixDisplay(accuracy2)
    
    """
        Print overall accuracy, precision, recall and display confusion matrix
    """
    print("Confusion Matrix: ")
    print(accuracy2)
    
    print("Precsion: ", '{:.3%}'.format(prec_recall[0]))
    print("Recall  : ", '{:.3%}'.format(prec_recall[1]))
    print("F1-score: ", '{:.3%}'.format(prec_recall[2]))
    print("Overall accuracy: ", '{:.3%}'.format(accuracy))
    
    disp.plot()
    plt.show()
    
    """Concatennating y_pred to daily_inputs to create daily output file"""
    daily_inputs_req = daily_inputs.iloc[:, 1:31]
    ypred = pd.DataFrame(y_pred, columns=["Y_PRED"])
    daily_outputs = pd.concat([daily_inputs_req.reset_index(drop = False), ypred], axis=1)
    
    print("--------------------------------------------------------------------")  
    
    return daily_outputs
    
if __name__ == '__main__':
    
    """
       Uploads data to train the network 
       Uploads daily data for prediction
    """
    inputs = pd.read_csv('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/school_shooter_db.csv')
    daily_inputs = pd.read_csv('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/daily_data.csv')
    feat = pd.read_csv('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/Features.csv')
  
    daily_outputs = daily_predict(inputs, daily_inputs)
    report(feat, daily_outputs)
    
    daily_outputs.to_csv('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/daily_output_data.csv')
    print("******Finished******")
