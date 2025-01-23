import pandas as pd
import numpy as np

class DecisionTreePAI:
    def __init__(self):
        self.class_mappings = {}
        self.num_classes = 0
        self.attributes = []
        
    def setup_classes(self, num_classes):
        """Initialize class structure based on user input"""
        self.num_classes = num_classes
        for attr_idx in range(len(self.attributes)):
            class_dict = {}
            for i in range(num_classes):
                class_name = f"class_{attr_idx+1}{i+1}"
                value = input(f"Enter value for {class_name} (attribute: {self.attributes[attr_idx]}): ")
                class_dict[class_name] = value
            self.class_mappings[self.attributes[attr_idx]] = class_dict

    def gini_index_df(self, target, feature, uniques):
        """Calculate Gini index and return detailed DataFrame"""
        gini_data = []
        weighted_gini = 0
        total_count = feature.count()
        data = pd.concat([pd.DataFrame(target.values.reshape((target.shape[0], 1))), feature], axis=1)
        
        class_distributions = {}
        for value in uniques:
            value_count = feature[feature == value].count()
            if value_count > 0:
                gini = 0
                class_dist = {}
                for class_val in np.unique(target):
                    class_count = data.iloc[:,0][(data.iloc[:,0] == class_val) & (data.iloc[:,1] == value)].count()
                    class_dist[class_val] = class_count / value_count if value_count > 0 else 0
                    if class_count > 0:
                        gini += (class_count / value_count) ** 2
                        
                gini = 1 - gini
                weighted_gini += gini * (value_count / total_count)
                
                gini_data.append({
                    'Value': value,
                    'Gini': gini,
                    'Sample_Count': value_count,
                    **class_dist
                })
                class_distributions[value] = class_dist
        
        gini_data.append({
            'Value': 'Weighted_Gini',
            'Gini': weighted_gini,
            'Sample_Count': total_count
        })
        
        return pd.DataFrame(gini_data), class_distributions, weighted_gini

    def calculate_pai(self, gini_results):
        """Calculate PAI for all attributes"""
        pai_data = []
        
        # Get all attribute combinations for PAI calculation
        for i in range(len(self.attributes)):
            for j in range(i + 1, len(self.attributes)):
                attr1 = self.attributes[i]
                attr2 = self.attributes[j]
                
                # Calculate PAI for each class combination
                for class_num in range(1, self.num_classes + 1):
                    class1 = f"class_{i+1}{class_num}"
                    class2 = f"class_{j+1}{class_num}"
                    
                    value1 = self.class_mappings[attr1][class1]
                    value2 = self.class_mappings[attr2][class2]
                    
                    # Get Gini values for the specific classes
                    gini1 = gini_results[attr1][0].loc[gini_results[attr1][0]['Value'] == value1]['Gini'].values[0]
                    gini2 = gini_results[attr2][0].loc[gini_results[attr2][0]['Value'] == value2]['Gini'].values[0]
                    
                    # Calculate PAI
                    pai = (gini1 * gini_results[attr1][2] + gini2 * gini_results[attr2][2]) / 2
                    
                    pai_data.append({
                        'Class_Number': class_num,
                        'Attribute1': attr1,
                        'Attribute2': attr2,
                        'Value1': value1,
                        'Value2': value2,
                        'PAI': pai
                    })
        
        return pd.DataFrame(pai_data)

    def fit(self, data, target_column):
        """Fit the decision tree and calculate all metrics"""
        self.attributes = [col for col in data.columns if col != target_column]
        
        # Get number of classes from user
        self.num_classes = int(input("Enter number of classes for PAI calculation: "))
        self.setup_classes(self.num_classes)
        
        # Calculate Gini index for each attribute
        gini_results = {}
        for attr in self.attributes:
            uniques = data[attr].unique()
            gini_df, class_dist, weighted_gini = self.gini_index_df(
                data[target_column],
                data[attr],
                uniques
            )
            gini_results[attr] = (gini_df, class_dist, weighted_gini)
            
        # Calculate PAI
        pai_df = self.calculate_pai(gini_results)
        
        return {
            'gini_results': gini_results,
            'pai_results': pai_df
        }