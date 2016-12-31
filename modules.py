'''
Has all the user-defined modules needed for the Titanic data set
'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import make_scorer
from sklearn import grid_search


# Use pandas dummies to change the variables
def change_categorical_features(dtFrme):
    # Convert the data-frame into dummies and then do the linear regression model on age
    modfd_ftFrme=pd.DataFrame(index=dtFrme.index)
    
    for col, col_data in dtFrme.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
    
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'Embarked' => 'Embarked_S' and 'Embarked_Q'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        modfd_ftFrme = modfd_ftFrme.join(col_data)
    
    return modfd_ftFrme

def shuffle_split_data(X, y):
    """ Shuffles and splits data into 70% training and 30% testing subsets,
        then returns the training and testing subsets. """

    # Shuffle and split the data
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.30,random_state=100); 
    
    # Assigning to a particular value of random state gives same result everytime we run the code.
    

    # Return the training and testing data subsets
    return X_train, y_train, X_cv, y_cv

def performance_metric(y_true, y_predict,cfm=False,scorer='accuracy'):
    """ Calculates and returns the total error between true and predicted values. 
        Here F1-score is chosen as performance metric
    """
    if scorer=='f1':
        score = f1_score(y_true,y_predict);
    elif scorer=='mse':
        from sklearn.metrics import mean_squared_error
        score = mean_squared_error(y_true,y_predict);
    elif scorer=='accuracy':
        from sklearn.metrics import accuracy_score
        score=accuracy_score(y_true,y_predict)
        
    if cfm==True:
        print confusion_matrix(y_true, y_predict,labels=[1,0]); # Labels should be mentioned here 1 indicates 'Survived'
    
    return score


def learning_curves(X_train, y_train, X_cv, y_cv,mdl='DT'):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing error rates for each model are then plotted. """
    
    print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . ."
    
    # Create the figure window
    fig = plt.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    # rint rounds the array elements to nearest integers
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes)-1)
    test_err = np.zeros(len(sizes)-1)

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes[1:]): # Ignore the first element 
            # print s
            # Setup a decision tree classifier so that it learns a tree with max_depth = depth
            if (mdl=='DT'):
                clf = DecisionTreeClassifier(max_depth = depth)
            elif (mdl=='RF'):
                clf=RandomForestClassifier(max_depth=depth)
            # Fit the learner to the training data
            clf.fit(X_train[:s], y_train[:s])
            
            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], clf.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_cv, clf.predict(X_cv))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes[1:], test_err, lw = 2, label = 'Testing')
        ax.plot(sizes[1:], train_err, lw = 2, label = 'Training')
        ax.legend(framealpha=0.8)
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('$F_1$-score')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    if (mdl=='DT'):
        fig.suptitle('Decision Tree Classifier Learning Performances', fontsize=18, y=1.03)
        fig.savefig('plots/depth_f1_vs_dataPoints_%s.png'%mdl )
    elif (mdl=='RF'):
        fig.suptitle('Random Forest Classifier Learning Performances', fontsize=18, y=1.03)
        fig.savefig('plots/depth_f1_vs_dataPoints_%s.png'%mdl)
    
    print 'Done....creating learning curves'
    fig.tight_layout()
    # fig.show()



def fit_model(X, y,clf,parameters):
    """ Tunes a decision tree classifier model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """
    
    # Make an appropriate scoring function
    # We can use either of the following two statements
    # Here we should use F1-score or accuracy and the F1-score should be higher
    scoring_function = make_scorer(performance_metric,greater_is_better=True);
        
    # Make the GridSearchCV object
    print 'Starting grid-search for classifier...'
    
    cross_validation = StratifiedKFold(y, n_folds=5)
    
    opt_clf = grid_search.GridSearchCV(clf, parameters,scoring=scoring_function, cv=cross_validation);
    
    
    # Fit the learner to the data to obtain the optimal model with tuned parameters
    opt_clf.fit(X, y);
    
    print opt_clf.grid_scores_
    print opt_clf.best_estimator_
    print 'Done with grid-search'
    # Return the optimal model
    return opt_clf.best_estimator_