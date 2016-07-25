'''
Plot some basic survival statistics for the titanic data
'''
print(__doc__)
import matplotlib.pyplot as plt

from titanic import data, outcomes
from visualizations import survival_stats
# Plot-1
# Plot how many men and women has survived

figS,axS=survival_stats(data, outcomes, 'Sex');

axS.set_xticklabels(['Male','Female']);

figS.savefig('plots/sex_category.png')

#plt.show()
# Plot - 2
# X-axis is the age  
figA,axA=survival_stats(data, outcomes, 'Age');


figA.savefig('plots/age_category.png')
#plt.clf()

# Plot - 3
figMA,axMA=survival_stats(data, outcomes, 'Age', ["Sex == 'male'"]);


figMA.savefig('plots/male_age_category.png')

# Plot - 4
figFA,axFA=survival_stats(data, outcomes, 'Age', ["Sex == 'female'"]);

figFA.savefig('plots/female_age_category.png')

# Plot - 5
figFare,axFare=survival_stats(data, outcomes, 'Fare');

figFare.savefig('plots/fare_category.png')


# Plot - 6
# X axis is the passenger class
figPC,axPC=survival_stats(data, outcomes, 'Pclass');
axPC.set_xticklabels(range(1,4));
figPC.savefig('plots/Pclass.png');

#Plot -7
figParch,axParch=survival_stats(data,outcomes,'Parch');

figParch.savefig('plots/Parch.png');


# Plot-8
figSibSp,axSibSp=survival_stats(data,outcomes,'SibSp');
figSibSp.savefig('plots/SibSp.png');

# Plot-9
figEm,axEm=survival_stats(data,outcomes,'Embarked');
axEm.set_xticklabels(['C','Q','S']);
figEm.savefig('plots/Embarked.png');


plt.show()