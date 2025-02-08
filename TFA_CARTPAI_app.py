import streamlit as st
import pandas as pd
import numpy as np
from random import randint
from scipy import stats
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

from sklearn.tree import DecisionTreeClassifier, export_text
import networkx as nx

class DecisionTreePAI:
    def __init__(self):
        self.class_mappings = {}
        self.num_classes = 0
        self.attributes = []
        self.tree_structure = {}
        self.categorical_mappings = {}
        self.reverse_mappings = {}
        
    def create_categorical_mappings(self, data):
        """Create mappings for categorical values to integers"""
        self.categorical_mappings = {}
        self.reverse_mappings = {}
        
        # Standard order for categorical values
        value_order = {
            'VL': 0,  # Very Low
            'L': 1,   # Low
            'M': 2,   # Medium
            'H': 3,   # High
            'VH': 4   # Very High
        }
        
        for column in self.attributes:
            unique_values = sorted(data[column].unique())
            mapping = {}
            reverse_mapping = {}
            
            # Sort values based on their level (VL, L, M, H, VH)
            def get_value_level(x):
                # Extract the level part (last part of the string)
                level = ''.join(c for c in x if c.isalpha())[1:]  # Skip first letter
                if level == 'VLOW':
                    return 'VL'
                elif level == 'LOW':
                    return 'L'
                elif level == 'MEDIUM':
                    return 'M'
                elif level == 'HIGH':
                    return 'H'
                elif level == 'VHIGH':
                    return 'VH'
                return level

            # Sort unique values based on their levels
            sorted_values = sorted(unique_values, key=lambda x: value_order.get(get_value_level(x), 999))
            
            # Create mappings
            for i, value in enumerate(sorted_values):
                mapping[value] = i
                reverse_mapping[i] = value
                
            self.categorical_mappings[column] = mapping
            self.reverse_mappings[column] = reverse_mapping
            
        return self.categorical_mappings

    def convert_to_numeric(self, data):
        """Convert categorical data to numeric using mappings"""
        numeric_data = data.copy()
        for column in self.attributes:
            numeric_data[column] = numeric_data[column].map(self.categorical_mappings[column])
        return numeric_data
        
    def auto_detect_classes(self, data):
        """Automatically detect classes for each attribute"""
        self.class_mappings = {}
        for attr in self.attributes:
            unique_values = sorted(data[attr].unique())
            class_dict = {}
            for i, value in enumerate(unique_values, 1):
                class_name = f"class_{self.attributes.index(attr)+1}{i}"
                class_dict[class_name] = value
            self.class_mappings[attr] = class_dict
        self.num_classes = max(len(values) for values in self.class_mappings.values())

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
                
                # Convert numeric value back to categorical for display
                original_value = self.reverse_mappings[feature.name][value]
                gini_data.append({
                    'Value': original_value,
                    'Numeric_Value': value,
                    'Gini': gini,
                    'Sample_Count': value_count,
                    **class_dist
                })
                class_distributions[original_value] = class_dist
        
        gini_data.append({
            'Value': 'Weighted_Gini',
            'Numeric_Value': None,
            'Gini': weighted_gini,
            'Sample_Count': total_count
        })
        
        return pd.DataFrame(gini_data), class_distributions, weighted_gini

    def calculate_pai(self, gini_results):
        """Calculate PAI for all attribute combinations and their weighted values"""
        pai_data = []
        cluster_pai = {}
        
        # Get all attribute combinations for PAI calculation
        for i in range(len(self.attributes)):
            for j in range(i + 1, len(self.attributes)):
                attr1 = self.attributes[i]
                attr2 = self.attributes[j]
                cluster_key = f"{attr1}-{attr2}"
                
                # Get all values for each attribute from Gini results
                values1 = gini_results[attr1][0][gini_results[attr1][0]['Value'] != 'Weighted_Gini']
                values2 = gini_results[attr2][0][gini_results[attr2][0]['Value'] != 'Weighted_Gini']
                
                cluster_pais = []
                
                # Calculate PAI for each combination of values
                for _, row1 in values1.iterrows():
                    for _, row2 in values2.iterrows():
                        value1 = row1['Value']
                        value2 = row2['Value']
                        gini1 = row1['Gini']
                        gini2 = row2['Gini']
                        
                        # Calculate individual PAI
                        pai = (gini1 * gini_results[attr1][2] + gini2 * gini_results[attr2][2]) / 2
                        cluster_pais.append(pai)
                        
                        pai_data.append({
                            'Attribute1': attr1,
                            'Attribute2': attr2,
                            'Value1': value1,
                            'Value2': value2,
                            'Gini1': gini1,
                            'Gini2': gini2,
                            'PAI': pai
                        })
                
                # Calculate weighted PAI for this cluster
                sample_counts1 = values1['Sample_Count'].sum()
                sample_counts2 = values2['Sample_Count'].sum()
                total_samples = sample_counts1 + sample_counts2
                
                weighted_pai = sum(cluster_pais) / len(cluster_pais) * (total_samples / (total_samples + 1))
                cluster_pai[cluster_key] = weighted_pai
        
        return pd.DataFrame(pai_data), pd.Series(cluster_pai, name='Weighted_PAI')

    def validate_gini_with_sklearn(self, data, target_column):
        """Validate Gini calculations against sklearn's implementation"""
        validation_results = {}
        
        numeric_data = self.convert_to_numeric(data)
        
        for attr in self.attributes:
            # Calculate our Gini
            uniques = numeric_data[attr].unique()
            our_gini_df, _, our_weighted_gini = self.gini_index_df(
                data[target_column],
                numeric_data[attr],
                uniques
            )
            
            # Calculate sklearn's Gini
            dt = DecisionTreeClassifier(max_depth=1, criterion='gini')
            dt.fit(numeric_data[attr].values.reshape(-1, 1), data[target_column])
            
            sklearn_gini = dt.tree_.impurity[0]
            
            validation_results[attr] = {
                'our_weighted_gini': our_weighted_gini,
                'sklearn_gini': sklearn_gini,
                'difference': abs(our_weighted_gini - sklearn_gini)
            }
            
        return pd.DataFrame(validation_results).T

    def plot_decision_tree(self, results):
        """Plot the decision tree with Gini and PAI values"""
        G = nx.Graph()
        
        # Create root node
        G.add_node("Root", pos=(0.5, 1))
        
        # Add attribute nodes
        x_positions = np.linspace(0, 1, len(self.attributes))
        for i, attr in enumerate(self.attributes):
            gini_value = results['gini_results'][attr][2]  # Get weighted Gini
            node_label = f"{attr}\nGini: {gini_value:.3f}"
            G.add_node(node_label, pos=(x_positions[i], 0.7))
            G.add_edge("Root", node_label)
            
            # Add value nodes for this attribute
            gini_df = results['gini_results'][attr][0]
            values = gini_df[gini_df['Value'] != 'Weighted_Gini']['Value']
            num_values = len(values)
            if num_values > 0:
                value_x_positions = np.linspace(x_positions[i]-0.1, x_positions[i]+0.1, num_values)
                for j, value in enumerate(values):
                    value_gini = gini_df[gini_df['Value'] == value]['Gini'].values[0]
                    value_label = f"{value}\nGini: {value_gini:.3f}"
                    G.add_node(value_label, pos=(value_x_positions[j], 0.4))
                    G.add_edge(node_label, value_label)
        
        # Add PAI values as edge labels
        pai_labels = {}
        for _, row in results['pai_results'].iterrows():
            edge = (f"{row['Attribute1']}\nGini: {results['gini_results'][row['Attribute1']][2]:.3f}", 
                f"{row['Attribute2']}\nGini: {results['gini_results'][row['Attribute2']][2]:.3f}")
            if edge not in pai_labels:
                pai_labels[edge] = []
            pai_label = f"{row['Value1']}-{row['Value2']}: {row['PAI']:.3f}"
            pai_labels[edge].append(pai_label)
        
        # Add weighted cluster PAI values
        for (attr1, attr2), weighted_pai in results['cluster_pai'].items():
            edge = (f"{attr1}\nGini: {results['gini_results'][attr1][2]:.3f}", 
                f"{attr2}\nGini: {results['gini_results'][attr2][2]:.3f}")
            if edge in pai_labels:
                pai_labels[edge].append(f"Weighted PAI: {weighted_pai:.3f}")
        
        # Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(15, 10))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold')
        
        # Add PAI values as edge labels
        edge_labels = {}
        for edge, pais in pai_labels.items():
            edge_labels[edge] = '\n'.join(pais)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Decision Tree with Gini and PAI Values")
        plt.axis('off')
        return plt

    def fit(self, data, target_column):
        """Fit the decision tree and calculate all metrics"""
        self.attributes = [col for col in data.columns if col != target_column]
        
        # Create mappings and convert data to numeric
        self.create_categorical_mappings(data)
        numeric_data = self.convert_to_numeric(data)
        
        # Automatically detect classes using numeric data
        self.auto_detect_classes(numeric_data)
        
        # Calculate Gini index for each attribute
        gini_results = {}
        for attr in self.attributes:
            uniques = numeric_data[attr].unique()
            gini_df, class_dist, weighted_gini = self.gini_index_df(
                data[target_column],
                numeric_data[attr],
                uniques
            )
            gini_results[attr] = (gini_df, class_dist, weighted_gini)
        
        # Calculate PAI and weighted cluster PAI
        pai_df, cluster_pai = self.calculate_pai(gini_results)
        
        # Validate Gini calculations
        gini_validation = self.validate_gini_with_sklearn(data, target_column)
        
        return {
            'gini_results': gini_results,
            'pai_results': pai_df,
            'cluster_pai': cluster_pai,
            'gini_validation': gini_validation,
            'categorical_mappings': self.categorical_mappings
        }


FACTOR_CLASSES = {
    'VHIGH': (0.90, float('inf')),
    'HIGH': (0.80, 0.90),
    'MEDIUM': (0.25, 0.80),
    'LOW': (0.01, 0.25),
    'VLOW': (float('-inf'), 0.01)
}

class ChemicalAnalysis:
    def __init__(self):
        
        self.chemical_elements = None
        self.factor_categories = FACTOR_CLASSES
        self.attributes = []  # Added attributes list
        self.num_factors = None  # Store the number of factors globally

    def validate_data(self, df):
        """Validate input dataframe"""
        if df.empty:
            raise ValueError("Empty dataframe provided")
        if df.shape[0] < 2:  # Header + at least one row
            raise ValueError("Insufficient data rows")
        if df.shape[1] < 2:  # Elements + at least one sample
            raise ValueError("Insufficient columns")
        return True
    
    def categorize_scores(self, scores):
        """Categorize factor scores with chemical element prefixes"""
        categorized = scores.copy()
        for idx, col in enumerate(scores.columns):
            prefix = chr(70 + idx)  # F, S, T, etc.
            categorized[col] = scores[col].apply(
                lambda x: f"{prefix}{self._get_category(x)}"
            )
        return categorized

    def _get_category(self, value):
        """Get category for a value based on FACTOR_CLASSES"""
        for category, (lower, upper) in self.factor_categories.items():
            if lower <= value < upper:
                return category
        return 'VLOW'

    def load_chemical_elements(self, df):
        try:
            # Ensure dataframe has content
            if df is None or df.empty or df.shape[0] < 2:
                raise ValueError("Invalid data structure")

            # Get header row (sample numbers)
            header = df.columns.tolist()[1:]  # Skip first column
           
            # Get chemical elements (first column, skip header)
            self.chemical_elements = df.iloc[0:, 0].tolist()
            
            # Extract sample data (skip first column and header row)
            data = df.iloc[0:, 1:].values
            

            # Create processed dataframe
            processed_df = pd.DataFrame(
                data=data,
                index=self.chemical_elements,
                columns=[f'Sample_{i+1}' for i in range(len(header))]
            )
            
            # Validate final structure
            if processed_df.empty:
                raise ValueError("Failed to process data")
                
            return processed_df  # Transpose for analysis
        
        except Exception as e:
            print(f"Error in data loading: {str(e)}")
            return pd.DataFrame()

    def preprocess_chemical_data(self, df):
        try:
            # Load and validate data
            processed_df = self.load_chemical_elements(df)
            if processed_df.empty:
                raise ValueError("Data preprocessing failed")
                
            # Scale the data
            scaler = StandardScaler()
            processed_df = processed_df.T
            
            scaled_data = scaler.fit_transform(processed_df)
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=processed_df.columns,
                index=processed_df.index
            )
            
            # Statistical tests
            chi_square, p_value = calculate_bartlett_sphericity(scaled_df)
            kmo_all, kmo_model = calculate_kmo(scaled_df)
            
            return scaled_df, {
                'bartlett': {'chi_square': chi_square, 'p_value': p_value},
                'kmo': kmo_model
            }
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return pd.DataFrame(), {}

    def perform_factor_analysis(self, scaled_df):
        try:
            # Initial analysis for scree plot
            fa_initial = FactorAnalyzer(rotation=None)
            fa_initial.fit(scaled_df.T)
            ev, v = fa_initial.get_eigenvalues()
            
            # Plot scree
            self.plot_scree(ev)
            
            # Set number of factors
            if self.num_factors is None:
                n_factors = len([x for x in ev if x > 1])
                print(f"Suggested number of factors: {n_factors}")
                self.num_factors = int(input("Enter number of factors to use: "))
            
            # Create factor analysis object and perform factor analysis
            fa = FactorAnalyzer(self.num_factors, rotation="varimax", method='minres', use_smc=True)
            fa.fit(scaled_df)

            FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=True,
                        method='minres', n_factors=self.num_factors, rotation='varimax',
                        rotation_kwargs={}, use_smc=True)

            # Get results
            loadings = pd.DataFrame(
                fa.loadings_,
                columns=[f'Factor_{i+1}' for i in range(self.num_factors)],
                index=scaled_df.columns
            )
            
            scores = pd.DataFrame(
                fa.transform(scaled_df),
                columns=[f'Factor_{i+1}' for i in range(self.num_factors)],
                index=scaled_df.index
            )
            
            return scores, loadings

        except Exception as e:
            print(f"Error in factor analysis: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
            
        
    def plot_scree(self, eigenvalues):
        """Plot scree plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.axhline(y=1, color='r', linestyle='--')
        plt.grid(True)
        plt.show()

    def generate_labels(self, categorized_scores):
        """
        Generate binary labels based on factor class levels for each column independently.
        Once a row is marked as 'Yes', it maintains that label even if subsequent factors
        would mark it as 'No'.
        
        Parameters:
        categorized_scores (pd.DataFrame): DataFrame with categorized scores for each factor
        
        Returns:
        pd.Series: Binary labels ('Yes'/'No') for each row
        """
        # Define class levels from highest to lowest
        class_levels = ['VHIGH', 'HIGH', 'MEDIUM', 'LOW', 'VLOW']
        
        # Initialize label array with 'No'
        n_samples = len(categorized_scores)
        labels = ['No'] * n_samples
        
        # Process each factor column independently
        for col in categorized_scores.columns:
            # Extract class for each sample (remove prefix F,S,T)
            factor_classes = categorized_scores[col].apply(
                lambda x: x[1:] if isinstance(x, str) else x  # Remove prefix
            )
            
            # Find all unique classes in this column
            unique_classes = set(factor_classes)
            
            # Find the maximum class level among the classes actually present in this column
            max_class = None
            for level in class_levels:
                if level in unique_classes:
                    max_class = level
                    break
            
            if max_class is not None:
                # Mark 'Yes' for rows where this factor has its maximum class
                # but don't override existing 'Yes' labels with 'No'
                for idx, class_val in enumerate(factor_classes):
                    if class_val == max_class:
                        labels[idx] = 'Yes'
                    # If this factor would mark it as 'No', we keep any existing 'Yes'
                    # (do nothing if it's already 'Yes')
        
        return pd.Series(labels, index=categorized_scores.index)

    def process_sheet(self, df):
        try:
            # Preprocess data
            scaled_df, tests = self.preprocess_chemical_data(df)
            
            # Store features
            self.attributes = scaled_df.columns.tolist()
        
            # Factor Analysis
            scores, loadings = self.perform_factor_analysis(scaled_df)
            
            # Categorize scores
            categorized = self.categorize_scores(loadings)
         
            # Generate labels
            labels = self.generate_labels(categorized)
            categorized['Label'] = labels
            
            return {
                'factor_analysis': {'scores': scores, 'loadings': loadings, 'tests': tests},
                'elements': self.chemical_elements,
                'categorized': categorized,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing sheet: {str(e)}")
            return None

    def analyze_excel(self, excel_path):
        """Process Excel file with chemical composition data"""
        try:
            # Read Excel file
            df = pd.read_excel(excel_path, sheet_name=0)  # Read only the first sheet      
            # Validate data structure
            if self.validate_data(df):
                return self.process_sheet(df)
            
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            return {}
        
def main():
    st.set_page_config(page_title="TFA & CARTPAI Analysis", page_icon=":bar_chart:")

    # Load logo (replace 'logo.png' with your actual logo path)
    st.image("TFA Logo.png", use_column_width=True) 

    # File upload
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Create an instance of ChemicalAnalysis
            analysis = ChemicalAnalysis()

            # Analyze the Excel file
            results_fa = analysis.analyze_excel(df)

            if results_fa:
                # Display Factor Loadings
                st.header("Factor Loadings")
                st.dataframe(results_fa['factor_analysis']['loadings'])

                # User input for manual categorization (optional)
                manual_categorization = st.checkbox("Manually Categorize Classes and Labels")

                # Download Factor Loadings
                st.download_button(
                    "Download Factor Loadings",
                    data=results_fa['factor_analysis']['loadings'].to_csv(index=True),
                    file_name="factor_loadings.csv",
                    mime="text/csv"
                )

                #if manual_categorization:
                    # Handle manual categorization logic here (not implemented)

                # Default categorization and Decision Tree output
                st.header("Decision Tree Output (Default Categorization)")

                # Create and fit the model
                df = pd.DataFrame(results_fa['categorized'])
                dt_pai = DecisionTreePAI()
                results = dt_pai.fit(df, 'Label')

                # Display Decision Tree (visual or textual)
                # You can use a library like graphviz or pydotplus for visual representation
                # Here's a basic textual representation:
                tree_text = export_text(DecisionTreeClassifier().fit(
                    df.drop('Label', axis=1), df['Label']
                ))
                st.text(tree_text)

                # Display PAI values
                st.header("PAI Values")
                st.dataframe(results['pai_results'])

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
