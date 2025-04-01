import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import DecisionTreeClassifier, export_text
import plotly.express as px
import plotly.graph_objects as go

# Define the factor classes for categorization
FACTOR_CLASSES = {
    'VHIGH': (0.90, float('inf')),
    'HIGH': (0.80, 0.90),
    'MEDIUM': (0.25, 0.80),
    'LOW': (0.01, 0.25),
    'VLOW': (float('-inf'), 0.01)
}

class ClassLevelPAI:
    def __init__(self):
        self.class_mappings = {}
        self.attributes = []
        self.categorical_mappings = {}
        self.reverse_mappings = {}
        self.class_levels = ['VHIGH', 'HIGH', 'MEDIUM', 'LOW', 'VLOW']
        
    def create_categorical_mappings(self, data):
        """Create mappings for categorical values to integers"""
        self.categorical_mappings = {}
        self.reverse_mappings = {}
        
        # Standard order for categorical values
        value_order = {
            'VLOW': 0,
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3,
            'VHIGH': 4
        }
        
        for column in self.attributes:
            unique_values = sorted(data[column].unique())
            mapping = {}
            reverse_mapping = {}
            
            # Sort values based on their level (remove first character which is the factor prefix)
            def get_value_level(x):
                return x[1:] if len(x) > 1 else x

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
        
    def gini_index_df(self, target, feature, uniques):
        """Calculate Gini index and return detailed DataFrame"""
        gini_data = []
        weighted_gini = 0
        total_count = len(feature)
        
        # Ensure target and feature are properly aligned
        data = pd.DataFrame({'target': target, 'feature': feature})
        
        class_distributions = {}
        for value in uniques:
            subset = data[data['feature'] == value]
            value_count = len(subset)
            
            if value_count > 0:
                gini = 0
                class_dist = {}
                
                # Count occurrences of each class value for this feature value
                for class_val in np.unique(target):
                    class_count = sum(subset['target'] == class_val)
                    class_proportion = class_count / value_count
                    class_dist[class_val] = class_proportion
                    gini += class_proportion ** 2
                
                gini = 1 - gini
                weighted_gini += gini * (value_count / total_count)
                
                # Convert numeric value back to categorical for display
                original_value = self.reverse_mappings.get(feature.name, {}).get(value, value)
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

    def calculate_class_level_pai(self, gini_results):
        """Calculate PAI values at class level (VLOW, LOW, MEDIUM, HIGH, VHIGH) across factors"""
        # Extract class levels from the data
        class_levels = set()
        for attr in self.attributes:
            for value in gini_results[attr][0]['Value']:
                if value != 'Weighted_Gini':
                    # Extract class level (remove first character which is the factor prefix)
                    level = value[1:] if len(value) > 1 else value
                    class_levels.add(level)
        
        # Initialize PAI data
        pai_data = {}
        for level in class_levels:
            pai_data[level] = {'factors': [], 'gini_values': [], 'weighted_gini': []}
        
        # Collect Gini values for each class level across factors
        for attr in self.attributes:
            attr_prefix = attr[0] if len(attr) > 0 else ''  # Get first character as prefix (F, S, T, etc.)
            
            for _, row in gini_results[attr][0].iterrows():
                if row['Value'] != 'Weighted_Gini':
                    value = row['Value']
                    # Extract class level (remove first character)
                    level = value[1:] if len(value) > 1 else value
                    
                    if level in pai_data:
                        pai_data[level]['factors'].append(attr)
                        pai_data[level]['gini_values'].append(row['Gini'])
                        pai_data[level]['weighted_gini'].append(gini_results[attr][2])
        
        # Calculate PAI for each class level
        pai_results = []
        for level, data in pai_data.items():
            if len(data['factors']) > 0:
                # Calculate PAI as average of (Gini * WeightedGini) for this class level
                pai_components = []
                for g, wg in zip(data['gini_values'], data['weighted_gini']):
                    pai_components.append(g * wg)
                
                pai = sum(pai_components) / len(data['factors'])
                
                pai_results.append({
                    'Class_Level': level,
                    'Factors': ', '.join(data['factors']),
                    'Factor_Count': len(data['factors']),
                    'Gini_Values': data['gini_values'],
                    'PAI': pai
                })
        
        return pd.DataFrame(pai_results)

    def fit(self, data, target_column):
        """Fit the model and calculate all metrics using class-level PAI approach"""
        self.attributes = [col for col in data.columns if col != target_column]
        
        # Create mappings and convert data to numeric
        self.create_categorical_mappings(data)
        numeric_data = self.convert_to_numeric(data)
        
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
        
        # Calculate class-level PAI
        pai_df = self.calculate_class_level_pai(gini_results)
        
        # Calculate total PAI
        total_pai = pai_df['PAI'].sum()
        
        return {
            'gini_results': gini_results,
            'pai_results': pai_df,
            'total_pai': total_pai,
            'categorical_mappings': self.categorical_mappings
        }

class ChemicalAnalysis:
    def __init__(self):
        self.chemical_elements = None
        self.factor_categories = FACTOR_CLASSES
        self.attributes = []
        self.num_factors = None

    def validate_data(self, df):
        """Validate input dataframe"""
        if df.empty:
            raise ValueError("Empty dataframe provided")
        if df.shape[0] < 2:
            raise ValueError("Insufficient data rows")
        if df.shape[1] < 2:
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
        """Load chemical elements from the dataframe"""
        try:
            # Ensure dataframe has content
            if df is None or df.empty or df.shape[0] < 2:
                raise ValueError("Invalid data structure")

            # Get header row (sample numbers)
            header = df.columns.tolist()[1:]  # Skip first column
           
            # Get chemical elements (first column)
            self.chemical_elements = df.iloc[:, 0].tolist()
            
            # Extract sample data (skip first column)
            data = df.iloc[:, 1:].values
            
            # Create processed dataframe
            processed_df = pd.DataFrame(
                data=data,
                index=self.chemical_elements,
                columns=[f'S{i+1}' for i in range(len(header))]
            )
            
            # Validate final structure
            if processed_df.empty:
                raise ValueError("Failed to process data")
                
            return processed_df.T  # Transpose for analysis
        
        except Exception as e:
            st.error(f"Error in data loading: {str(e)}")
            return pd.DataFrame()

    def preprocess_chemical_data(self, df):
        """Preprocess the chemical data"""
        try:
            # Load and validate data
            processed_df = self.load_chemical_elements(df)
            if processed_df.empty:
                raise ValueError("Data preprocessing failed")
                
            # Scale the data
            scaler = StandardScaler()
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
            st.error(f"Error in preprocessing: {str(e)}")
            return pd.DataFrame(), {}

    def perform_factor_analysis(self, scaled_df, num_factors=3):
        """Perform factor analysis with the specified number of factors"""
        try:
            # Set number of factors
            self.num_factors = num_factors
            
            # Create factor analysis object and perform factor analysis
            fa = FactorAnalyzer(self.num_factors, rotation="varimax", method='minres', use_smc=True)
            fa.fit(scaled_df)

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
            st.error(f"Error in factor analysis: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def generate_labels(self, categorized_scores):
        """Generate binary labels based on factor class levels"""
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
        
        return pd.Series(labels, index=categorized_scores.index)

    def process_sheet(self, df, num_factors=3):
        """Process a single sheet/depth of data"""
        try:
            # Preprocess data
            scaled_df, tests = self.preprocess_chemical_data(df)
            
            # Store features
            self.attributes = scaled_df.columns.tolist()
        
            # Factor Analysis
            scores, loadings = self.perform_factor_analysis(scaled_df, num_factors)
            
            # Categorize loadings
            categorized = self.categorize_scores(loadings)
         
            # Generate labels
            labels = self.generate_labels(categorized)
            categorized['Label'] = labels
            
            # Calculate PAI
            pai_model = ClassLevelPAI()
            pai_results = pai_model.fit(categorized, 'Label')
            
            return {
                'factor_analysis': {'scores': scores, 'loadings': loadings, 'tests': tests},
                'elements': self.chemical_elements,
                'categorized': categorized,
                'labels': labels,
                'pai_results': pai_results
            }
            
        except Exception as e:
            st.error(f"Error processing sheet: {str(e)}")
            return None

    def analyze_multiple_depths(self, dfs, num_factors=3):
        """Process multiple sheets/depths and compare results"""
        all_results = []
        
        for i, df in enumerate(dfs):
            st.write(f"Processing Depth {i+1}...")
            results = self.process_sheet(df, num_factors)
            if results:
                results['depth'] = i + 1
                all_results.append(results)
        
        return all_results
        
    def plot_factor_loadings(self, loadings, depth_label=""):
        """Plot factor loadings heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Factor Loadings {depth_label}')
        plt.tight_layout()
        return plt

    def plot_pai_comparison(self, all_results):
        """Plot PAI values across depths for each class level"""
        # Extract PAI values from all results
        pai_data = []
        for result in all_results:
            depth = result['depth']
            for _, row in result['pai_results']['pai_results'].iterrows():
                pai_data.append({
                    'Depth': depth,
                    'Class_Level': row['Class_Level'],
                    'PAI': row['PAI']
                })
        
        # Create DataFrame for plotting
        pai_df = pd.DataFrame(pai_data)
        
        # Create plot
        fig = px.bar(
            pai_df, 
            x='Class_Level', 
            y='PAI', 
            color='Depth',
            barmode='group',
            title='PAI Values by Class Level Across Depths',
            labels={'PAI': 'PAI Value', 'Class_Level': 'Class Level'}
        )
        
        return fig

    def plot_class_level_stability(self, all_results):
        """Plot stability of class-level PAI values across depths"""
        # Extract PAI values
        pai_data = []
        for result in all_results:
            depth = result['depth']
            for _, row in result['pai_results']['pai_results'].iterrows():
                pai_data.append({
                    'Depth': depth,
                    'Class_Level': row['Class_Level'],
                    'PAI': row['PAI']
                })
        
        # Create DataFrame
        pai_df = pd.DataFrame(pai_data)
        
        # Calculate statistics for each class level
        class_stats = pai_df.groupby('Class_Level').agg({
            'PAI': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Rename columns
        class_stats.columns = ['Class_Level', 'PAI_Mean', 'PAI_StdDev', 'PAI_Min', 'PAI_Max']
        
        # Calculate coefficient of variation
        class_stats['PAI_CV'] = class_stats['PAI_StdDev'] / class_stats['PAI_Mean']
        
        # Sort by mean PAI value
        class_stats = class_stats.sort_values('PAI_Mean', ascending=False)
        
        # Create plot
        fig = go.Figure()
        
        # Add bars for mean PAI
        fig.add_trace(go.Bar(
            x=class_stats['Class_Level'],
            y=class_stats['PAI_Mean'],
            name='Mean PAI',
            error_y=dict(
                type='data',
                array=class_stats['PAI_StdDev'],
                visible=True
            )
        ))
        
        # Add CV as text
        for i, row in enumerate(class_stats.itertuples()):
            fig.add_annotation(
                x=row.Class_Level,
                y=row.PAI_Mean + row.PAI_StdDev + 0.01,
                text=f"CV: {row.PAI_CV:.2f}",
                showarrow=False
            )
        
        fig.update_layout(
            title='Class Level PAI Stability Across Depths',
            xaxis_title='Class Level',
            yaxis_title='Mean PAI Value',
            barmode='group'
        )
        
        return fig

def main():
    st.set_page_config(page_title="Multi-Depth Chemical Analysis", page_icon=":bar_chart:", layout="wide")
    
    st.title("Multi-Depth Chemical Analysis with PAI")
    st.write("""
    This application performs factor analysis on chemical composition data across multiple depths,
    and calculates class-level PAI (Predictive Attribute Interaction) values.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel file with multiple sheets (one per depth)", type=["xlsx"])
    
    if uploaded_file is not None:
        # Load data from all sheets
        try:
            # Read all sheets
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            
            if len(sheet_names) == 0:
                st.error("No sheets found in the Excel file.")
                return
            
            st.success(f"Found {len(sheet_names)} sheets (depths) in the file.")
            
            # Number of factors selection
            num_factors = st.slider("Select number of factors for analysis", min_value=1, max_value=5, value=3)
            
            # Create instance of ChemicalAnalysis
            analysis = ChemicalAnalysis()
            
            # Load data from each sheet
            dfs = []
            for sheet in sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                dfs.append(df)
            
            # Process all depths
            if st.button("Analyze All Depths"):
                with st.spinner("Analyzing all depths..."):
                    all_results = analysis.analyze_multiple_depths(dfs, num_factors)
                
                if not all_results:
                    st.error("Error processing data.")
                    return
                
                # Display results for each depth in expandable sections
                st.header("Results by Depth")
                
                for result in all_results:
                    depth = result['depth']
                    
                    with st.expander(f"Depth {depth}"):
                        # Display factor loadings
                        st.subheader(f"Factor Loadings - Depth {depth}")
                        st.dataframe(result['factor_analysis']['loadings'])
                        
                        # Plot factor loadings
                        fig_loadings = analysis.plot_factor_loadings(result['factor_analysis']['loadings'], f"Depth {depth}")
                        st.pyplot(fig_loadings)
                        
                        # Display categorized scores
                        st.subheader(f"Categorized Factor Scores - Depth {depth}")
                        st.dataframe(result['categorized'])
                        
                        # Display Gini results
                        st.subheader(f"Gini Results - Depth {depth}")
                        for attr, (gini_df, _, weighted_gini) in result['pai_results']['gini_results'].items():
                            st.write(f"{attr} - Weighted Gini: {weighted_gini:.4f}")
                            st.dataframe(gini_df)
                        
                        # Display PAI results
                        st.subheader(f"PAI Results - Depth {depth}")
                        st.dataframe(result['pai_results']['pai_results'])
                        st.write(f"Total PAI: {result['pai_results']['total_pai']:.4f}")
                
                # Comparative analysis
                st.header("Comparative Analysis Across Depths")
                
                # Plot PAI comparison
                st.subheader("PAI Values by Class Level Across Depths")
                fig_pai = analysis.plot_pai_comparison(all_results)
                st.plotly_chart(fig_pai)
                
                # Plot class level stability
                st.subheader("Class Level PAI Stability")
                fig_stability = analysis.plot_class_level_stability(all_results)
                st.plotly_chart(fig_stability)
                
                # Download combined results
                st.header("Download Results")
                
                # Create Excel file with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Summary sheet
                    summary_data = []
                    for result in all_results:
                        depth = result['depth']
                        for _, row in result['pai_results']['pai_results'].iterrows():
                            summary_data.append({
                                'Depth': depth,
                                'Class_Level': row['Class_Level'],
                                'PAI': row['PAI']
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='PAI_Summary', index=False)
                    
                    # Individual depth sheets
                    for result in all_results:
                        depth = result['depth']
                        # Factor loadings
                        result['factor_analysis']['loadings'].to_excel(writer, sheet_name=f'D{depth}_Loadings')
                        # Categorized scores
                        result['categorized'].to_excel(writer, sheet_name=f'D{depth}_Categories')
                        # PAI results
                        result['pai_results']['pai_results'].to_excel(writer, sheet_name=f'D{depth}_PAI', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="Download All Results (Excel)",
                    data=output,
                    file_name="multi_depth_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Please check that your Excel file is properly formatted with one depth per sheet.")

if __name__ == "__main__":
    main()
