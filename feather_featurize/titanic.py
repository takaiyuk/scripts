"""
- input/train.csv and input/test.csv are taken from kaggle titanic competition:
https://www.kaggle.com/c/titanic/data
"""

import os
import pandas as pd

from feather_featurize import Feature, get_arguments, generate_features

Feature.dir = 'features'

def grouping_age(age):
    if age<13:
        return 'child'
    elif age<25:
        return 'youth'
    elif age<60:
        return 'adult'
    elif age>=60:
        return 'elder'
    else:
        return "None" 

class FamilySize(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1
        
class AgeGroup(Feature):
    def create_features(self):
        self.train['age_group'] = train['Age'].map(grouping_age)
        self.test['age_group'] = test['Age'].map(grouping_age)


if __name__ == '__main__':
    if not os.path.exists(Feature.dir):
        os.mkdir(Feature.dir)
        
    args = get_arguments()

    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    generate_features(globals(), args.force)
