'''
Cleans the data for any missing values. Especially with Age
Both in test and train data set
@author: Bala, bgsquare@gmail.com
'''
import pandas as pd
import numpy as np
from collections import Counter
from titanic import full_data,test_data
from visualizations import survival_stats
from sklearn.metrics import mean_squared_error


def fit_model_dtr(X, y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """
    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import make_scorer
    from sklearn import grid_search
    #print X,y
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor();
    
    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    
    # Make an appropriate scoring function
    # We can use either of the following two statements
    scoring_function = make_scorer(mean_squared_error,greater_is_better=False);
    
        
    # Make the GridSearchCV object
   
    reg = grid_search.GridSearchCV(regressor, parameters,scoring=scoring_function);
    
    
    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y);
    
    # print reg.grid_scores_
    # print reg.best_estimator_

    # Return the optimal model
    return reg.best_estimator_

def imputeAge(data,train=True):
    '''
    Imputes the Age data based on the mean of the title
    '''
    data=pd.DataFrame(data)
    # print train
    name_titles_set=set(['Master','Miss','Mrs','Mr']);
    
    title=pd.DataFrame(columns=['Title'])
    # print data.Name
    title['Title']=data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
    
    def other(x):
        if x not in name_titles_set:
            return 'Others'
        else:
            return x
    
    title['Title']=title['Title'].apply(other); # Applies the function other on each of the data element
    
        
    data=pd.concat([data, title], axis = 1)
    if train==True:
        outcomes=data['Survived']
        data=data.drop(['Survived'],axis=1)
        
        figT,axT=survival_stats(data, outcomes, 'Title');
    
        axT.set_xticklabels(['Master','Miss','Mrs','Mr','Others']);
    
    
     
    # Fill the age with the mean of the age group based on title
    name_titles_set.add('Others'); # Add is used in sets
    for tle in name_titles_set:
        # print 'Mean of age for Title', tle,data['Age'][data['Title']==tle].mean()
        # print 'before',data['Age'][data['Title']==tle]
        data['Age'][data['Title']==tle]=data['Age'][data['Title']==tle].fillna(data['Age'][data['Title']==tle].mean())
        # print 'after',data['Age'][data['Title']==tle]
    
    if train==True:
        # Append back the outcomes
        data=pd.concat([data,outcomes],axis=1)
        
        data.to_csv('data/imputedAge_train.csv',index=False)
        figT.savefig('plots/title_category_train.png')
    else:
        data.to_csv('data/imputedAge_test.csv',index=False)
        
    
    return data


def imputeAge_regression(data,train=True):
    '''
    Impute the Age based using regression models Linear/Decision Tree Regression
    '''
    from sklearn.linear_model import LinearRegression
   
   
    full_data=pd.DataFrame(data); # full_data is the un-modified data
    
    dropped_features=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
    data=data.drop(dropped_features,axis=1)
    if train==True:
        # Drop the Survived set
        data=data.drop(['Survived'],axis=1)
    
    
    # Convert the data-frame into dummies and then do the linear regression model on age
    modified_data=pd.DataFrame(index=data.index)
    
    for col, col_data in data.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'Embarked' => 'Embarked_S' and 'Embarked_Q'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        modified_data = modified_data.join(col_data)
    
    
    # Separate the data-set into two data-frames, where the age needs to be predicted
    df_age_na=modified_data[modified_data['Age'].isnull()]; # This has to be predicted
    df_age=modified_data[modified_data['Age'].notnull()];
    
    age_outcomes=df_age.Age
    # Drop the Age from the data-frame
    df_age=df_age.drop('Age',axis=1)
    df_age_na=df_age_na.drop('Age',axis=1)
    
    # Using Linear Regression model
    LR_model=LinearRegression()
    LR_model.fit(df_age,age_outcomes)
    # print LR_model.coef_, LR_model.intercept_
    LR_age_prediction= LR_model.predict(df_age_na)
    
    # Now use the DecisionTreeRegressor
    
    dt_reg=fit_model_dtr(df_age,age_outcomes)
    
    DTR_age_prediction=dt_reg.predict(df_age_na)
    
    predict_age_df=pd.DataFrame(DTR_age_prediction,columns=['Age'],index=df_age_na.index)
    
    # Add predicted data to the data-frame where there were null
    
    # Combine first fills the NaN values based on the index
    full_data_dtr=full_data.combine_first(predict_age_df)
    # Replace all the missing data on the 'Embarked' as 'S=Southampton'
    full_data_dtr.Embarked=data.Embarked.fillna('S')
    if train==True:
        full_data_dtr.to_csv('data/imputedAge_train_regression.csv',index=False)
    else:
        full_data_dtr.to_csv('data/imputedAge_test_regression.csv',index=False)
    
    return full_data_dtr




def remove_outliers(data,outcomes): # We should also remove the data from outcomes
    '''
    Input data should be in a categorical form
    '''
    
    outliers=[];
    for feature in data.keys():
    
       
        
        Q1 = np.percentile(data[feature],25,axis=0);
        
        
       
        Q3 = np.percentile(data[feature],75,axis=0);
        
        # print Q1,Q3,feature
        
        
        IQR=Q3-Q1; # This is the inter-quartile range
        step = 1.5*IQR; # Based on John Tukey's method, it can be 1.5 or 3 (which is more conservative)
        
        # Display the outliers
        print 'Q1 and Q3 scores for the feature {0:s} are {1:2.2f} and {2:2.2f} respectively'.format(feature,Q1,Q3)
        # print "Data points considered outliers for the feature '{}':".format(feature)
        
        # display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
        outliers_for_feature= list(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))].index)
        # append only the indices that are already not part of outliers
        
        outliers.extend(outliers_for_feature) # .extend adds the individual items of list instead of adding(appending) entire list

    # Remove the outliers, if any were specified
    counter_dict= dict(Counter(outliers)) # Counts how many times each index has appeared as the outlier for every feature.
    
    # Key is the index of the data and value is  number of times it appears as outliers
    count_outlier_df=pd.DataFrame.from_dict(counter_dict,orient='index');
    count_outlier_df.columns=['Count']; # Name the colum count
    # display(count_outlier_df.sort_index())
    # Print the outlier where the count is more than 1
    # print 'Outliers that are present in more than one feature'
    # print count_outlier_df[count_outlier_df['Count']>1]
    outliers=list(set(outliers)) # set() removes any duplicate values in the list. Then convert back to list 
    print 'Number of outliers for all features are {0}'.format(len(outliers));
    print 'Size of original data is', data.shape
    good_data = data.drop(data.index[outliers]).reset_index(drop = True)
    good_outcomes=outcomes.drop(outcomes.index[outliers]).reset_index(drop = True)
    print 'Sanitized data size is', good_data.shape
    assert(len(set(outliers))==data.shape[0]-good_data.shape[0])
    
    return good_data,good_outcomes

# imputeAge(full_data,True)
# imputeAge(test_data,False)

imputeAge_regression(full_data,True)