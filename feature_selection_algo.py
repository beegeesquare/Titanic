'''
This script is for feature selection.
(1) Sequential forward search algorithm
(2) Sequential backward search algorithm
(3) Randomized algorithms, such as Simulated Annealing or Genetic algorithms (TODO) 
@author: Bala, bgsquare@gmail.com
'''

print __doc__;

from sklearn.cross_validation import train_test_split 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# Start the feature selection process

def forward_search(X,y):
    '''
    Uses forward search algorithm for feature selection process.
    Take each feature at a time and compute the score. Pick the feature with maximum score and keep adding,
    features. This is greedy algorithm
    '''
    print 'Starting forward search feature selection algorithm....'
    
    # Initialize the feature set
    good_features=set([]); # These set gets appended as we select the feature to be added
    
    score_history=[]; # Keeps the history of scores as we add the features
    
    all_features=X.columns; # Initialize this to all features
    
    
    while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]: # The second condition compares whether the current score is greater than previous score
    
        scores=[];
        
        for feature in all_features:
            
            if feature not in good_features:
                selected_features=list(good_features)+[feature]
        
                # Get the data-frame for the selected 
                print selected_features
                X_fs=X[selected_features]; # This should give the data frame for the selected features
                            
                # Shuffle and split the data
                X_train_f,X_test_f,y_train_f,y_test_f =train_test_split(X_fs, y, test_size=0.25,random_state=13); 
                
                # Apply the learning algorithm
               
                reg=RandomForestClassifier(random_state=13);
                
                reg.fit(X_train_f,y_train_f)
                
                y_pred_f=reg.predict(X_test_f);
                
                # Report the score for the selected feature
                
                feature_score=f1_score(y_test_f,y_pred_f)
                
                scores.append((feature_score,feature))
                
                print 'F_1-score for the feature {0:s} added: {1:.4f}'.format(feature,feature_score);
            
            
        
        print sorted(scores)
        
        
        # good features is a set, so we use add instead of append
        good_features.add(sorted(scores)[-1][1]); # Pick the last element in the sorted list. Element in the list is two-tuple
        
        print 'Selected feature(s) are', good_features
        
        score_history.append(sorted(scores)[-1]); # Append the tuple
        
        print 'this is score history', score_history
        
        
        
    # Outside the while loop
    # Remove the last feature that was added. 
    # Because of that feature the score went down i.e., the while loop's second condition broke.
    # print score_history[-1][1]
    good_features.remove(score_history[-1][1])
    # print good_features
    # print score_history
    
    return list(good_features)

def backward_search(X,y):
    '''
    Uses backward search algorithm for feature selection process.
    Starting with all features and removing each feature at a time. This is greedy algorithm
    '''
    
    print 'Starting backward search feature selection algorithm....'
    
    # Initialize the feature set
    good_features=set(list(X.columns)); # Here we start with all features
    
    # print good_features
    
    score_history=[]; # Keeps the history of scores as we remove the features
    
    all_features=X.columns; # Initialize this to all features
    
    
    while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]: # The second condition compares whether the current score is greater than previous score
    
        scores=[];
        
        for feature in all_features:
            
            if feature in good_features:
                selected_features=list(good_features)
                selected_features.remove(feature); # Remove the feature 
                
                # print good_features
                
                # Get the data-frame for the selected 
                print selected_features
                X_fs=X[selected_features]; # This should give the data frame for the selected features
                            
                # Shuffle and split the data
                X_train_f,X_test_f,y_train_f,y_test_f =train_test_split(X_fs, y, test_size=0.25,random_state=13); 
                
                # Apply the learning algorithm
               
                reg=RandomForestClassifier(random_state=13);
                
                reg.fit(X_train_f,y_train_f)
                
                y_pred_f=reg.predict(X_test_f);
                
                # Report the score for the selected feature
                
                feature_score=f1_score(y_test_f,y_pred_f)
                
                scores.append((feature_score,feature))
                
                print 'F_1-score for the feature {0:s} removed: {1:.4f}'.format(feature,feature_score);
            
            
        
        print sorted(scores)
        
        
        # good features is a set, so we use add instead of append
        # Pick the last element in the sorted list, this is the feature that needs to be removed . Element in the list is two-tuple 
        good_features.remove(sorted(scores)[-1][1]); 
        
        print 'Selected feature(s) are', good_features
        
        score_history.append(sorted(scores)[-1]); # Append the tuple
        
        print 'this is score history, after removing the feature', score_history
        
        # break # For testing
       
    # Outside the while loop
    # Put back the last feature that was removed. 
    # Because of that feature being removed the score went down i.e., the while loop's second condition broke.
    # print score_history[-1][1]
    good_features.add(score_history[-1][1])
    # print good_features
    # print score_history
    
    return list(good_features)

