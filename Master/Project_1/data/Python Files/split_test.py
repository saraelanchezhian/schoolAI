import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from classifier import *

def tester(test_percent):
    """
        Reads data from the school shooter database csv file created from the list of following sources: 
        1. https://www.fbi.gov/file-repository/stats-services-publications-school-shooter-school-shooter/view
        2. https://www.chds.us/ssdb/data-map/
        3. https://doi.org/10.1016/S0272-7358(99)00055-0
        4. Additional cases simulated by completely randomizing covariate assignment in order to cover various scenarios
    """ 
    inputs = pd.read_csv('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/school_shooter_db.csv')
    
    
    """
        Data split into training and testing batches with a 80/20 or 70/30 or 60/40 proportion.
    """
    X_train, X_test = train_test_split(inputs, test_size=test_percent, shuffle=True, stratify=inputs['Y'])
    if test_percent == 0.2:
        splitdisp = '80/20'
    if test_percent == 0.3:
        splitdisp = '70/30'
    if test_percent == 0.4:
        splitdisp = '60/40'
    
    """
        Initializing threat level classifier model to train data.
    """
    model = ThreatLevelClassifier(X_train)
    model.train()
    
    """
        Invoke model to predict and classfiy output for test data.
    """
    y_pred = model.predict(X_test)
    y_test = X_test.iloc[:,32]

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
    print("Confusion Matrix for ", splitdisp, " spilt: ")
    print(accuracy2)
    
    print("Precsion for ", splitdisp, " spilt: ", '{:.3%}'.format(prec_recall[0]))
    print("Recall   for ", splitdisp, " spilt: ", '{:.3%}'.format(prec_recall[1]))
    print("F1-score for ", splitdisp, " spilt: ", '{:.3%}'.format(prec_recall[2]))
    print("Overall accuracy for ", splitdisp, " spilt: ", '{:.3%}'.format(accuracy))
    
    disp.plot()
    plt.show()
    
    print("--------------------------------------------------------------------")    
        
if __name__ == '__main__':
        
    """Testing for 80/20 split"""
    tester(0.2)
    
    """Testing for 70/30 split"""
    tester(0.3)
    
    """Testing for 60/40 split"""
    tester(0.4)
    
    print("******Finished******")
    
    
    
    