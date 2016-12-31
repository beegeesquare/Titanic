'''
This script uses Tree based machine learning models such as:
Model 1. Decision Tree Classifier
Model 1a. Decision Tree Classifier, with grid-search
@author: Bala, bgsquare@gmail.com
'''
print __doc__;


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import matplotlib.pyplot as plt
# User-defined
from titanic import full_data,test_data
from clean_data import imputeAge, imputeAge_regression
from visualizations import visualize_tree
import modules

plt.style.use('ggplot')

# Update your data-set with the age being imputed (estimated)

useTitle=False;

# impute the missing values in ticket-fare





print 'Using title {0:}'.format(useTitle)

if useTitle==True:
    dropped_features=['PassengerId', 'Name', 'Ticket', 'Cabin','Title'];
    full_data=imputeAge(full_data,train=True)
    test_data=imputeAge(test_data, train=False)
else:
    dropped_features=['PassengerId', 'Name', 'Ticket', 'Cabin'];
    full_data=imputeAge_regression(full_data,train=True)
    test_data=imputeAge_regression(test_data, train=False)
    
    
outcomes=full_data['Survived']; # Get the Y that you are trying to predict

train_data=pd.DataFrame(full_data.drop(['Survived'],axis=1))

train_data.Embarked=train_data.Embarked.fillna('S');

feature_cols = list(train_data.columns);

print "Feature columns ({} total features):\n{}".format(len(train_data.columns), list(train_data.columns));

# Step-0 Encode the categorical features

train_data=train_data.drop(dropped_features,axis=1)

# New variable are added based on the categorical feature
train_data=modules.change_categorical_features(train_data)


full_test_data=pd.DataFrame(test_data); # Make a copy of full test-data
test_data=test_data.drop(dropped_features,axis=1)
# Use pandas dummies to make the categorical data same as the train_data
test_data=modules.change_categorical_features(test_data)


print 'End of all the pre-processing...'
print 'Starting Model 1.....'

# Test shuffle_split_data

try:
    X_train, y_train, X_cv, y_cv = modules.shuffle_split_data(train_data, outcomes); 
    print "Successfully shuffled and split the data!"
except:
    print "Something went wrong with shuffling and splitting the data."


clf_A = DecisionTreeClassifier(random_state=13); # Creates a decision tree classifier, goes full depth default max_depth=
# Step-1: Fit the model with the data

clf_A.fit(X_train, y_train);


# Visualize the Decision Tree graph:

visualize_tree(clf_A,list(X_train.columns),filename='dt_train');
y_cv_predict=clf_A.predict(X_cv)

print 'F1-score for Decision tree using the CV-set is {0:.4f}'.format(modules.performance_metric(y_cv,y_cv_predict,True))
print 'Accuracy score for Decision tree using the CV-set is {0:.4f}'.format(accuracy_score(y_cv,y_cv_predict))


# Step-2: Given the model, predict Y using test set


Y_predict= pd.DataFrame(clf_A.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)
Y_predict.to_csv('data/DT_test_result_1.csv',index=False)


# Before we start the grid search for the classifier, see the performance of learning for different depths
# Plot the graphs for varying test and train sets. But here training set is split


# Plot some learning-curves for variation of different data-sizes
modules.learning_curves(X_train, y_train, X_cv, y_cv)

print 'Model 1a Decision tree...using Grid search'
# Create a decision tree classifier object
clf = DecisionTreeClassifier();
# Set up the parameters we wish to tune
parameters = {'max_depth':range(1,50),'criterion':('gini','entropy')}
try:
    opt_clf = modules.fit_model(X_train, y_train,clf,parameters)
    print "Successfully fit a Optimal Decision tree model, using grid search!"
    
except:
    print "Something went wrong with fitting a model."

print 'F1-score for Optimal Decision tree using the CV data-set is {0:.4f}'.format(f1_score(y_cv,opt_clf.predict(X_cv)))
print 'Accuracy score for Optimal Decision tree using the CV data-set is {0:.4f}'.format(accuracy_score(y_cv,opt_clf.predict(X_cv)))


# Visualize this on the tree (this outside the above try function, since there is problem using this there)
visualize_tree(opt_clf, list(X_train.columns), filename='opt_dt_train')

# Take the opt_clf and use it on test-set
Y_predict=opt_clf.predict(X_cv);

print 'F1-score for Optimal Decision tree using the CV data-set is {0:.4f}'.format(f1_score(y_cv,Y_predict))


# Using this model and test-data set, predict the output of the passengers
# We don't know the outputs
Y_predict=pd.DataFrame(opt_clf.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)

Y_predict.to_csv('data/opt_DT_test_result_1.csv',index=False)

