import numpy as np
import pandas as pd


# RMS Titanic data visualization code 


# from IPython.display import display
#matplotlib inline

# Load the dataset
in_file = 'data/train.csv'; # Training data for titanic
full_data = pd.read_csv(in_file); # Reads the CSV file into a dataframe

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived'];
data = full_data.drop('Survived', axis = 1); # Survived is not a feature so remove it from the columns

# Load the test test
in_file_test='data/test.csv';

# Test set does not have any 'Survived' column.


# Convert csv file to panadas data frame
test_data=pd.read_csv(in_file_test);


def start():
    print 'Starting Titanic data exploration....';
    
    # Print the first few entries of the RMS Titanic data
    print(full_data.head());
    
    
    # Number of passengers
    print 'Total number of passengers {}'.format(len(outcomes));
    
    # Show the new dataset with 'Survived' removed
    print(data.head());

    return

#start()