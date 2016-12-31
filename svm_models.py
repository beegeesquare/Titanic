'''
This script uses Support Vector based models such as:
Model 1. Support vector machines
@author: Bala, bgsquare@gmail.com
'''

print __doc__;

from sklearn import svm
from sklearn import grid_search
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from titanic import full_data,test_data
from clean_data import imputeAge
from visualizations import pca_results

plt.style.use('ggplot')

# Update your data-set with the age being imputed (estimated)

full_data=imputeAge(full_data,train=True)
test_data=imputeAge(test_data,train=False)

# When predicting use the same features as the training set and also do the label encoding
full_test_data=pd.DataFrame(test_data); # Make a copy of full test-data



outcomes=full_data['Survived']; # Get the Y that you are trying to predict
train_data=pd.DataFrame(full_data.drop(['Survived'],axis=1))

feature_cols = list(train_data.columns);

print "Feature columns ({} total features):\n{}".format(len(train_data.columns), list(train_data.columns));

print 'Titanic data set is already split into training and test sets'


# Instead of using label encoding use the pd.dummies here

# Pre-process the data, i.e., convert the categorical data 
def preprocess_features(X,dropped_features):
    ''' Preprocesses the Titanic data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    for i in dropped_features:   X=X.drop(i, axis = 1);
    
    
    output = pd.DataFrame(index = X.index); # Initializes the number of rows in X which is passengers
    
    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        # Fill in the missing data with the mean
        if col_data.dtype!=object: # That means the column data is numerical
            #print 'doing this'
            X[col]= X[col].fillna(X[col].median()); # This won't happen for Age, since age has already been imputed
        
        elif col=='Embarked':
            X[col]= X[col].fillna('S'); # Assume they all are embarked at 'S'=South Ampton
        
        if col_data.dtype==col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'Age' => 'Age_male' and 'Age_female'
            # Example: 'Embarked' => 'Embarked_S'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

print 'End of all the pre-processing...'

# For making the prediction features such as PassengerID, name, Ticket, Fare,Cabin and Embarked does not matter
# From the data, let's drop these features'
# dropped_features=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked','Title'];
# dropped_features=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin','Title'];
dropped_features=['PassengerId', 'Name', 'Ticket', 'Cabin','Fare','Title'];

train_data=preprocess_features(train_data, dropped_features)
test_data=preprocess_features(test_data, dropped_features)

print "Features after processing columns ({} total features):\n{}".format(len(train_data.columns), list(train_data.columns));

# There needs to be another step for splitting and training, but data already has been splitted
clf_A = svm.SVC(random_state=13); # Creates a SVC 
                                                 
# Step-1: Fit the model with the data

clf_A.fit(train_data, outcomes);

Y_predict=clf_A.predict(train_data)

print 'F1-score for Support Vector Classifier model using the entire data-set is {0:.4f}'.format(f1_score(outcomes,Y_predict))

from sklearn.cross_validation import train_test_split 

def shuffle_split_data(X, y):
    """ Shuffles and splits data into 70% training and 30% testing subsets,
        then returns the training and testing subsets. """

    # Shuffle and split the data
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.30,random_state=100); 
    # Assigning to a particular value of random state gives same result everytime we run the code.
    

    # Return the training and testing data subsets
    return X_train, y_train, X_cv, y_cv

def performance_metric(y_true, y_predict):
    """ Calculates and returns the total error between true and predicted values. 
        Here F1-score is chosen as performance metric
    """
    error = f1_score(y_true,y_predict);

    return error

# Test shuffle_split_data
try:
    X_train, y_train, X_cv, y_cv = shuffle_split_data(train_data, outcomes); # CV =cross=validation set
    print "Successfully shuffled and split the data!"
except:
    print "Something went wrong with shuffling and splitting the data."

print 'F1-score for Support Vector Classifier model using the cross-validation data-set is {0:.4f}'.format(f1_score(y_cv,clf_A.predict(X_cv)))

def learning_curves(X_train, y_train, X_cv, y_cv):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing error rates for each model are then plotted. """
    
    print "Creating learning curve graphs for different kernels...."
    
    kernels = ['linear','poly','rbf','sigmoid'] # Other things you can vary, 'C':[1,10,50, 100],'degree':[3,5]; 
    # kernels=['linear', 'poly']
    # Create the figure window
    fig = plt.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    # rint rounds the array elements to nearest integers
    sizes = np.rint(np.linspace(1, len(X_train), 25)).astype(int)
    train_err = np.zeros(len(sizes)-1)
    test_err = np.zeros(len(sizes)-1)

    # Create four different models based on Kernel/C/degree
    for k, krnl in enumerate(kernels):
        print krnl
        for i, s in enumerate(sizes[1:]): # Ignore the first element 
            print s
            # Setup a Support Vector classifier so that it learns a tree with max_depth = depth
            
            clf = svm.SVC(kernel = krnl)
           
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
        ax.set_title('Kernel = %s'%(krnl))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('$F_1$-score')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    
    fig.suptitle('Support Vector Classifier Learning Performances', fontsize=18, y=1.03)
    fig.savefig('plots/kernel_f1_vs_dataPoints_.png')
    
    
    print 'Done....creating learning curves'
    fig.tight_layout()
    # fig.show()
    
    return
# Plot some learning-curves for variation of different data-sizes
# learning_curves(X_train, y_train, X_cv, y_cv)


poly_clf=svm.SVC(kernel='poly');
poly_clf.fit(train_data,outcomes)

print 'F1-score for Polynomial Kernal SVM {0:.4f}'.format(f1_score(y_cv,poly_clf.predict(X_cv)));


Y_predict=pd.DataFrame(poly_clf.predict(test_data),columns=['Survived']);
Y_predict=pd.concat([full_test_data['PassengerId'],Y_predict],axis=1)


Y_predict.to_csv('data/poly_svm_test_result.csv',index=False)
    

'''
parameters = {'kernel':['poly'], 'C':[1,10,50, 100],'degree':[3,5]}; # Choosing two Kernels 
f1_scorer = make_scorer(performance_metric); # greater_is_better=True by default

grid_obj = grid_search.GridSearchCV(clf_A, parameters, scoring=f1_scorer);
grid_obj.fit(X_train,y_train)

opt_svm=grid_obj.best_estimator_
print opt_svm
print grid_obj.best_params_
print grid_obj.best_score_


print 'F1-score for optimal SVC model using entire data set is {0:.4f}'.format(f1_score(outcomes,opt_svm.predict(train_data)))
print 'F1-score for optimal SVC model using CV set is {0:.4f}'.format(f1_score(y_cv,opt_svm.predict(X_cv)))
'''


# This 
    
'''   
# Take the PCA and transform it into two-dimensions to see the SVM
pca = PCA(n_components=train_data.shape[1]); # If n_components is not specified then the default value is used

pca.fit(train_data)

print pca.mean_
print pca.components_

# Generate PCA results plot
pca_results = pca_results(train_data, pca)
print 'Explained Varience'
print pca_results['Explained Variance']
# Cummulative sum is the sum of all elements before that and including that element $y_k=\sum_{i=1}^{k}x_i$
print pca_results['Explained Variance'].cumsum()

pca = PCA(n_components=2);

# Apply a PCA transformation the good data
pca.fit(train_data);
reduced_data = pca.transform(train_data);

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
figTran=pd.scatter_matrix(reduced_data, color='b',alpha = 0.3, figsize = (6,6), diagonal = 'kde');

# Use the reduced data on the SVM

print reduced_data.head()

clf = svm.SVC(kernel='linear', C=50.0);
clf.fit(reduced_data, outcomes);

# get the separating hyperplane
w = clf.coef_[0]; # Coefficient
a = -w[0] / w[1]; #y interscept
#xx = np.linspace(-5, 5);

xx=np.linspace(reduced_data['Dimension 1'].min(), reduced_data['Dimension 1'].max())
yy = a * xx - (clf.intercept_[0]) / w[1]; # This is decision boundary
# plot the parallels to the separating hyperplane that pass through the
# support vectors
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin; # Two support vectors
yy_up = yy - a * margin; #

# plot the line, the points, and the nearest vectors to the plane
#plt.figure(1, figsize=(4, 3));
plt.clf(); # Here clf is clear figure not be confused with classifer (clf)
#plt.plot(xx, yy, 'k-',linewidth=2);
#plt.plot(xx, yy_down, 'k--');
#plt.plot(xx, yy_up, 'k--');
#print clf.support_vectors_[:, 0]
#print clf.support_vectors_[:, 1]
#print clf.n_support_

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
plt.scatter(reduced_data['Dimension 1'], reduced_data['Dimension 2'], c=outcomes, zorder=10, cmap=plt.cm.Paired)
#for i in range(len(clf.support_vectors_[:, 0])):
#    plt.annotate('Support vectors', xy=(clf.support_vectors_[:, 0][i], clf.support_vectors_[:, 1][i]), xytext=(3, 4),
#            arrowprops=dict(arrowstyle="->"));
maxDataPoint_args=np.argmax(np.array(reduced_data),axis=0); # Axis=0 gives the maximum column wise
maxDataPoint1=np.array(reduced_data)[maxDataPoint_args[0],:];
maxDataPoint2=np.array(reduced_data)[maxDataPoint_args[1],:];
# plt.annotate('A', xy=(maxDataPoint1[0],maxDataPoint1[1]), xytext=(maxDataPoint1[0]+0.05,maxDataPoint1[1]+0.05));
plt.axis('tight');
plt.xlabel('$X_{1}$');
plt.ylabel('$X_{2}$');
'''

plt.show()
