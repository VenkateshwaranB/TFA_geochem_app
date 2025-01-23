import pandas as pd
import pytest
from src.components.decision_tree import DecisionTreePAI

def test_gini_index_df():
    data = {
        'Factor 1': ['FVLOW', 'FVHIGH', 'FVHIGH', 'FVLOW'],
        'Label': ['NO', 'YES', 'YES', 'NO']
    }
    df = pd.DataFrame(data)
    dt_pai = DecisionTreePAI()
    dt_pai.attributes = ['Factor 1']
    gini_df, class_dist, weighted_gini = dt_pai.gini_index_df(df['Label'], df['Factor 1'], df['Factor 1'].unique())

    assert len(gini_df) == 3  # Check number of unique values
    assert 'Weighted_Gini' in gini_df['Value'].values  # Check if weighted Gini is calculated

def test_calculate_pai():
    data = {
        'Factor 1': ['FVLOW', 'FVHIGH', 'FVHIGH', 'FVLOW'],
        'Label': ['NO', 'YES', 'YES', 'NO']
    }
    df = pd.DataFrame(data)
    dt_pai = DecisionTreePAI()
    dt_pai.attributes = ['Factor 1']
    dt_pai.class_mappings = {'Factor 1': {'class_11': 'FVLOW', 'class_12': 'FVHIGH'}}
    dt_pai.num_classes = 2
    gini_results = {
        'Factor 1': (pd.DataFrame({'Value': ['FVLOW', 'FVHIGH', 'Weighted_Gini'], 'Gini': [0.5, 0.5, 0.5]}), {}, 0.5)
    }
    pai_df = dt_pai.calculate_pai(gini_results)

    assert len(pai_df) == 1  # Check number of PAI calculations
    assert pai_df['PAI'].iloc[0] == 0.5  # Check if PAI is calculated correctly

if __name__ == "__main__":
    pytest.main()