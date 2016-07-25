'''
This script determines the accuracy scores for the intituive models.

Model 1: Assume all the passengers have survived
Model 2: If a passenger is female, predict as survived
Model 3: If a passenger is female and a male passenger with age < 10, predict as survived
Model 4: Uses multiple features, by random selection, with a aim to improve the 
        accuracy at least 80%
'''
print(__doc__)


import pandas as pd
import numpy as np

from titanic import data,outcomes
from visualizations import accuracy_stats

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return (truth == pred).mean()*100
    
    else:
        # Return the percentage of accuracy
        print "Number of predictions does not match number of outcomes!"
        return 0


nbr_passengers=len(outcomes);
print 'Starting Model 1....'
# Assume all the passengers have survived
# Below line creates a dataframe with  
predictions = pd.Series(np.ones(nbr_passengers, dtype = int)); # Argument create an array of size=number of passengers
accuracyModel1= accuracy_score(outcomes, predictions); 

# Append each scores for the intutive models used
print 'Accuracy score for model 1:{0}'.format(accuracyModel1); 
accuracyScore=[accuracyModel1];

print 'Starting Model 2....'

def predictions_2(data):
    """ Model with one feature: 
          - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows(): # iterrows() give index and Series
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex']=='male':
            predictions.append(0); # Assuming, males have not survived
        else:
            predictions.append(1); # Females have survived
        
    
    # Return our predictions
    return pd.Series(predictions)

predictions=predictions_2(data);

accuracyModel2=accuracy_score(outcomes, predictions);

accuracyScore.append(accuracyModel2); # Update the accuracy scores 

print 'Accuracy score for model 2:{0}'.format(accuracyModel2); 

print 'Starting Model 3....'

def predictions_3(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        
        if (passenger['Sex']=='male') :
            if (passenger['Age']<10):
                predictions.append(1); 
            else:
                predictions.append(0); # Assuming, males with age greater than 10 did not survive
        else:
            predictions.append(1); # All females have survived and males with ages less than or equal 10 have survived
        
    # Return our predictions
    return pd.Series(predictions)

predictions=predictions_3(data);

accuracyModel3=accuracy_score(outcomes, predictions);

accuracyScore.append(accuracyModel3); # Update the accuracy scores 

print 'Accuracy score for model 3:{0}'.format(accuracyModel3); 

print 'Starting Model 4....'
def predictions_4(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if (passenger['Sex']=='male') :
            if (passenger['Age']<10):
                predictions.append(1); 
            elif (passenger['SibSp']>4):
                predictions.append(0);
            elif (passenger['Parch']>3):
                predictions.append(0);
            else:
                predictions.append(0);
        else:
            if (passenger['SibSp']>4):
                predictions.append(0);
            elif (passenger['Parch']>3):
                predictions.append(0);
            elif (passenger['Age']<=30):
                predictions.append(1);
            else:
                predictions.append(1);
            
    # Return our predictions
    return pd.Series(predictions)

predictions = predictions_4(data);

accuracyModel4= accuracy_score(outcomes, predictions);

accuracyScore.append(accuracyModel4); # Update the accuracy scores 

print 'Accuracy score for model 3:{0}'.format(accuracyModel4); 

accuracy_stats(accuracyScore,True);