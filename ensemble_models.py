'''
This script uses Tree based machine learning models such as:
Model 2. Random Forest Classifier without Grid search
Model 2a. Random Forest Classifier with Grid search
@author: Bala, bgsquare@gmail.com
'''

print __doc__;

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.cross_validation import StratifiedKFold

import pandas as pd

import matplotlib.pyplot as plt

from titanic import full_data,test_data
from clean_data import imputeAge, imputeAge_regression
from visualizations import visualize_tree
import modules


plt.style.use('ggplot')

# Update your data-set with the age being imputed (estimated)

useTitle=False;

print 'Using title {0:}'.format(useTitle)

if useTitle==True:
    dropped_features=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin','Title'];
    full_data=imputeAge(full_data,train=True)
    test_data=imputeAge(test_data, train=False)
else:
    dropped_features=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'];
    full_data=imputeAge_regression(full_data,train=True)
    test_data=imputeAge_regression(test_data, train=False)
    

outcomes=full_data['Survived']; # Get the Y that you are trying to predict
train_data=pd.DataFrame(full_data.drop(['Survived'],axis=1))
# train_data.Embarked=train_data.Embarked.fillna('S');
feature_cols = list(train_data.columns);
print "Feature columns ({} total features):\n{}".format(len(train_data.columns), list(train_data.columns));


train_data=train_data.drop(dropped_features,axis=1)
# New variable are added based on the categorical feature
train_data=modules.change_categorical_features(train_data)
full_test_data=pd.DataFrame(test_data); # Make a copy of full test-data

test_data=test_data.drop(dropped_features,axis=1)
# Use pandas dummies to make the categorical data same as the train_data
test_data=modules.change_categorical_features(test_data)


'''
probable_featurs= list(test_data.columns)
# Do the feature-selection algorithm

good_features_ffs=feature_selection_algo.forward_search(train_data,outcomes);
print good_features_ffs
good_features_bfs=feature_selection_algo.backward_search(train_data,outcomes);
print good_features_bfs
# Drop the features that are not in good_features_ list
ForwardSearch=False;

if ForwardSearch==True:
    for ftr in probable_featurs:
        if ftr not in good_features_ffs:
            train_data=train_data.drop(ftr,axis=1)
            test_data=test_data.drop(ftr,axis=1)
else:
    for ftr in probable_featurs:
        if ftr not in good_features_bfs:
            train_data=train_data.drop(ftr,axis=1)
            test_data=test_data.drop(ftr,axis=1)

print 'Features used after feature selection are {}'.format(list(train_data.columns));
'''


print 'End of all the pre-processing...'

# Random-Forest Classifier

print 'Starting Model 2..... Random Forest classifier'

# Test shuffle_split_data

try:
    X_train, y_train, X_cv, y_cv = modules.shuffle_split_data(train_data, outcomes); 
    print "Successfully shuffled and split the data!"
except:
    print "Something went wrong with shuffling and splitting the data."


n_estimators = 30
clf_B = RandomForestClassifier(random_state=13,n_estimators=n_estimators); # Creates a Random Forest Classifier                                                 


# Step-1: Fit the model with the data

clf_B.fit(X_train, y_train);


print clf_B

y_cv_predict=clf_B.predict(X_cv)

print 'Accuracy scores for Random Forest classifier using CV data-set is {0:.4f}'.format(modules.performance_metric(y_cv,y_cv_predict,True,scorer='accuracy'))


# Use this on test-set
Y_predict= pd.DataFrame(clf_B.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)
Y_predict.to_csv('data/RF_test_result_1_fs.csv',index=False)

# Plot some learning-curves for variation of different data-sizes
modules.learning_curves(X_train, y_train, X_cv, y_cv,mdl='RF')
plt.show()

print 'Starting grid-search for Random Forest classifiers...'
clf = RandomForestClassifier();
# Set up the parameters we wish to tune
parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10),'n_estimators':(25,50,75,100), 'criterion':('gini','entropy')}

# Test fit_model using training data-set and cross-validation set (which is X_train here)
try:
    opt_clf_rf = modules.fit_model(X_train, y_train,clf,parameters)
    print "Successfully fit a Optimal Random Forest, using grid search!"
    
except:
    print "Something went wrong with fitting a model."


# Take the opt_clf and use it on test-set
Y_predict=opt_clf_rf.predict(X_cv);
print 'Accuracy score for Optimal Random forest using the CV data-set is {0:.4f}'.format(accuracy_score(y_cv,Y_predict))


# Using this model and test-data set, predict the output of the passengers
# We don't know the outputs
Y_predict=pd.DataFrame(opt_clf_rf.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)

Y_predict.to_csv('data/opt_RF_test_result_1_fs.csv',index=False)

'''
# Gradient Boost Classifier

clf_C=GradientBoostingClassifier()

parameters = {'loss':('deviance','exponential'),'n_estimators':(200,210,230,250)}

try:
    opt_clf_gbc = modules.fit_model(X_train, y_train,clf_C,parameters)
    print "Successfully fit a Optimal Gradient Boost classifier, using grid search!"
    
except:
    print "Something went wrong with fitting a model."


# Take the opt_clf and use it on test-set
Y_predict=opt_clf_gbc.predict(X_cv);
print 'F1-score for Optimal Gradient Boost classifier using the CV data-set is {0:.4f}'.format(f1_score(y_cv,Y_predict))

Y_predict=pd.DataFrame(opt_clf_gbc.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)

Y_predict.to_csv('data/opt_gbc_test_result_1.csv',index=False)


# Adaboost Classifier

clf_D=AdaBoostClassifier()

clf_D.fit(X_train,y_train)

Y_predict=clf_D.predict(X_cv)
print 'F1-score for AdaBoost classifier using the CV data-set is {0:.4f}'.format(f1_score(y_cv,Y_predict))

Y_predict=pd.DataFrame(clf_D.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)

Y_predict.to_csv('data/opt_Ada_test_result_1.csv',index=False)
'''
