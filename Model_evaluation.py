### Model evaluation function

# Output the confusion matrix and ROC.
# You don't need to modify codes here.


import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)


def Model_evaluation(y_test,y_pred):
    '''
       Evaluate the model performance.
       You don't need to modify codes here.

       Input:
       X_test   : Test sample explainable variables
       y_test   : Test sample explained variable
       y_pred   : The predicted value of your model
       logreg   : You constructed model

       Output:
        Confusion matrix
        ROC
        (You will learn what these are in the later lecture.)
       '''

    # Get the confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)

    # Get the ROC curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r',linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return confusion_matrix