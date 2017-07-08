'''
This script uses Machine Learning Models such as:
Model 1. Decision Tree Classifier
Model 2. Support Vector Machines (TODO)
Model 3. Neural Networks (TODO)
Model 4: Guassian Navie Bayes (TODO)
@author: Bala, bgsquare@gmail.com
'''
print __doc__;

from titanic import data,outcomes,test_data
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import subprocess
import warnings
#from sklearn.cross_validation import cross_val_score
# from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

feature_cols = list(data.columns);

print "Feature columns ({} total features):\n{}".format(len(data.columns), list(data.columns));


# For making the prediction features such as PassengerID, name, Ticket, Fare,Cabin and Embarked does not matter
# From the data, let's drop these features'
droped_features=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'];


# Pre-process the data, i.e., convert the categorical data 
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    for i in droped_features:   X=X.drop(i, axis = 1);
    
    # Initialize new output DataFrame
    
    # Drop all the passengers where data is missing
    # X=X.dropna();
    # print X.index
    #try:
    #    Y_new=pd.Series(Y,index=X.index); # update Y after dropping the new indices
    #except:
    #    pass
    # Drop the the outcomes from that a
    # Update outcomes
    #Y_new
    output = pd.DataFrame(index = X.index); # Initializes the number of rows in X which is passengers
    
    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        # Fill in the missing data with the mean
        if col_data.dtype!=object: # That means the column data is numerical
            #print 'doing this'
            X[col]= X[col].fillna(X[col].mean());
        
        if col_data.dtype==col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'Age' => 'Age_male' and 'Age_female'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output


# Before we preprocess the data


X=preprocess_features(data);

print "Processed feature columns ({} total features):\n{}".format(len(X.columns), list(X.columns))

print 'Starting Model 1.....'

clf_A = DecisionTreeClassifier(random_state=13); # Creates a decision tree classifier
# Step-1: Fit the model with the data

clf_A.fit(X, outcomes);
# Step-2: Given the model, predict the values of Y for X
Y_predict= clf_A.predict(X);

# Step-3: Calculate the F1-score for the predicted values on the training set
# F1-score: 2*recall*precision/(recall+precision);
# recall= 
print 'Accuracy-score on the training set using DT classifier is {0:.4f}'.format(accuracy_score(outcomes,Y_predict));

# Visualize the Decision Tree graph:
def visualize_tree(tree,features_list):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    command = ["dot", "-Tpng", "plots/dt_titanic.dot", "-o", "plots/dt_titanic.png"];
    try:
        with open("plots/dt_titanic_old.dot", 'w') as f:
            export_graphviz(tree, out_file=f, feature_names=features_list,filled=True, rounded=True,  special_characters=True);
       
    except:
        # Creates a directory if the plots directory does not exista
        subprocess.call(["mkdir", "plots"],shell=True);
        with open("plots/dt_titanic_old.dot", 'w') as f:
            export_graphviz(tree, out_file=f, feature_names=features_list,filled=True, rounded=True,  special_characters=True);

    try:
        subprocess.check_call(command,shell=True);
    except:
        warnings.warn("Could not run dot, ie graphviz, to "
             "produce visualization. Do it manually on terminal (such as cygwin)")


# Takes the decision tree and the list of features as inputs
visualize_tree(clf_A,list(X.columns));

# Now using test set, find the the F1-score
# Import the test set and preprocess the data 

X_test=preprocess_features(test_data);
Y_predict_test= clf_A.predict(X_test); # This will be numpy array (row vector)

Y_predict_df=pd.DataFrame(index=X_test.index);
Y_predict_df=Y_predict_df.join(test_data['PassengerId']);

Y_predict_df=Y_predict_df.join(pd.DataFrame(Y_predict_test,columns=['Prediction']));
# Rename the column that was predicted

# print Y_predict_df
Y_predict_df.to_csv(path_or_buf='data/dtModel.csv', index=False);