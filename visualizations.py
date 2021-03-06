import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import subprocess
import warnings

def filter_data(data, condition):
    """
    Remove elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=
    
    Example: ["Sex == 'male'", 'Age < 18']
    """

    field, op, value = condition.split(" ")
    
    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")
    
    # get booleans for filtering
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else: # catch invalid operation codes
        raise Exception("Invalid comparison operator. Only >, <, >=, <=, ==, != allowed.")
    
    # filter data and outcomes
    data = data[matches].reset_index(drop = True)
    return data

def survival_stats(data, outcomes, key, filters = []):
    """
    Print out selected statistics regarding survival, given a feature of
    interest and any number of filters (including no filters)
    """
    
    # Check that the key exists
    if key not in data.columns.values :
        print "'{}' is not a feature of the Titanic data. Did you spell something wrong?".format(key)
        return False

    # Return the function before visualizing if 'Cabin' or 'Ticket'
    # is selected: too many unique categories to display
    if(key == 'Cabin' or key == 'PassengerId' or key == 'Ticket'):
        print "'{}' has too many unique categories to display! Try a different feature.".format(key)
        return False

    # Merge data and outcomes into single dataframe
    all_data = pd.concat([data, outcomes], axis = 1)
    # print all_data.head()
    # Apply filters to data
    for condition in filters:
        all_data = filter_data(all_data, condition)

    # Create outcomes DataFrame
    all_data = all_data[[key, 'Survived']]
    
    plt.style.use('ggplot');
    figS=plt.figure();
    axS=figS.add_subplot(111);
    
    # Create plotting figure
    
    #plt.figure(figsize=(8,6))

    # 'Numerical' features
    if(key == 'Age' or key == 'Fare'):
        
        # Remove NaN values from Age data
        all_data = all_data[~np.isnan(all_data[key])]
        
        # Divide the range of data into bins and count survival rates
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value
        
        # 'Fares' has larger range of values than 'Age' so create more bins
        if(key == 'Fare'):
            bins = np.arange(0, all_data['Fare'].max() + 20, 20)
        if(key == 'Age'):
            bins = np.arange(0, all_data['Age'].max() + 10, 10)
            
        
        # Overlay each bin's survival rates
        nonsurv_vals = all_data[all_data['Survived'] == 0][key].reset_index(drop = True)
        surv_vals = all_data[all_data['Survived'] == 1][key].reset_index(drop = True)
        # print key
        # print nonsurv_vals
        # print surv_vals
        colors=['red','green'];
        labels=['Did not survive', 'Survived'];
        nonsurv_vals=np.array(nonsurv_vals)
        surv_stats=[nonsurv_vals,np.array(surv_vals)];
        axS.hist(surv_stats, bins = bins, alpha = 0.6,histtype='bar',stacked=True, color=colors,label=labels)
        
        '''
        axS.hist(surv_vals, bins = bins, alpha = 0.6,
                 color = 'green', label = 'Survived',histtype='bar')
        axS.hist(nonsurv_vals, bins = bins, alpha = 0.6,
                 color = 'red', label = 'Did not survive',histtype='bar');
        '''
        # Add legend to plot
        axS.set_xlim(0, bins.max())
        axS.legend(framealpha = 0.8)
    
    # 'Categorical' features
    else:
       
        # Set the various categories
        if(key == 'Pclass'):
            values = np.arange(1,4)
        if(key == 'Parch' or key == 'SibSp'):
            values = np.arange(0,np.max(data[key]) + 1)
        if(key == 'Embarked'):
            values = ['C', 'Q', 'S']
        if(key == 'Sex'):
            values = ['male', 'female']

        if (key=='Title'):
            values=['Master','Miss','Mrs','Mr','Others']
        
        # Create DataFrame containing categories and count of each
        
        frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Survived','NSurvived'))
        for i, value in enumerate(values):
            # print len(all_data[(all_data['Survived'] == 1) & (all_data[key] == value)])
            frame.loc[i] = [value, \
                   len(all_data[(all_data['Survived'] == 1) & (all_data[key] == value)]), \
                   len(all_data[(all_data['Survived'] == 0) & (all_data[key] == value)])]

        # Set the width of each bar
        bar_width = 0.4
        # print frame.head()
        # Display each category's survival rates
        for i in np.arange(len(frame)):
            nonsurv_bar = plt.bar(i-bar_width, frame.loc[i]['NSurvived'], width = bar_width, color = 'r')
            surv_bar = plt.bar(i, frame.loc[i]['Survived'], width = bar_width, color = 'g')

            #axS.set_xticks(np.arange(len(frame)), values)
            axS.set_xticks(np.arange(len(frame)));
            axS.set_xlabel(values);
            axS.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

    # Common attributes for plot formatting
    axS.set_xlabel(key);
    axS.set_ylabel('Number of Passengers')
    if len(filters)==0:
        axS.set_title('Passenger Survival Stats With \'%s\' Feature'%(key))
    else:
        axS.set_title('Passenger Survival Stats With \'%s\' Feature,'%(key))
        #axS.text('Filters used %s' %(filters));
    #plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        print "Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0))

    
    return figS,axS
    
def accuracy_stats(scores,intuitive):
    '''
    Plots the scores for each model. Intuituve=True/false
    Uses ggplot style
    '''
    plt.style.use('ggplot');
    fig,axes=plt.subplots();
    
    plt.plot(scores,'-ro',linewidth=2);
    
    if intuitive==True:
        axes.set_title('Accuracy Scores for intuitive models');
    else:
        axes.set_title('Accuracy Scores for machine learning models');
    
    axes.set_xticks(range(len(scores)));
    tickmarks=['Model '+str(mdidx+1) for mdidx in range(len(scores))];
    axes.set_xlim(-0.25,len(scores)-1+0.25);
    axes.set_xticklabels(tickmarks);
    axes.set_xlabel('Model');
    axes.set_ylabel('Percentage of accuracy score');
    # plt.show()  

def visualize_tree(tree,features_list,filename='dt'):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    command = ["dot", "-Tpng", "plots/dt_titanic.dot", "-o", "plots/dt_titanic.png"];
    try:
        with open("plots/%s_titanic.dot"%filename, 'w') as f:
            export_graphviz(tree, out_file=f, feature_names=features_list,filled=True, rounded=True,  special_characters=True);
       
    except:
        # Creates a directory if the plots directory does not exista
        subprocess.call(["mkdir", "plots"],shell=True);
        with open("plots/%s_titanic.dot"%filename, 'w') as f:
            export_graphviz(tree, out_file=f, feature_names=features_list,filled=True, rounded=True,  special_characters=True);

    try:
        subprocess.check_call(command,shell=True);
    except:
        warnings.warn("Could not run dot, ie graphviz, to "
             "produce visualization. Do it manually on terminal (such as cygwin)")

def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)


    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)
