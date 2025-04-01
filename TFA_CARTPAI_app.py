import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import DecisionTreeClassifier, export_text
import plotly.express as px
import plotly.graph_objects as go
import base64

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
        """Calculate Paleo Affinity Index (PAI) values at class level (VLOW, LOW, MEDIUM, HIGH, VHIGH) across factors"""
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
            title='Paleo Affinity Index (PAI) Values by Class Level Across Depths',
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
            title='Class Level Paleo Affinity Index (PAI) Stability Across Depths',
            xaxis_title='Class Level',
            yaxis_title='Mean PAI Value',
            barmode='group'
        )
        
        return fig

# Function to add a banner image
# Function to add a responsive banner with compact height
def add_banner(image_path=None):
    if image_path:
        try:
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"Error loading banner image: {e}")
    else:
        # Default banner with text
        st.markdown(
            """
            <div style="background-color:#4A5783; padding:10px; border-radius:10px; text-align:center;">
                <h1 style="color:white;">TFA & CARTPAI Analysis</h1>
                <h3 style="color:#E0E0E0;">Multi-Depth Chemical Analysis with Paleo Affinity Index (PAI)</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
# Function to create a downloadable sample data file
def get_sample_data():
    # Create a sample dataframe that matches the expected format
    sample_data_1 = pd.DataFrame({
        'Elements': ['SiO2', 'Al2O3', 'TiO2', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'MnO', 'P2O5', 'LOI'],
        'S1': [63.26, 13.36, 0.67, 5.79, 2.98, 2.74, 2.02, 3.28, 0.07, 0.12, 5.73],
        'S2': [67.16, 12.19, 0.63, 4.66, 2.57, 2.79, 2.21, 2.91, 0.05, 0.11, 5.08],
        'S3': [67.98, 11.87, 0.64, 4.41, 2.50, 2.85, 2.07, 2.95, 0.06, 0.14, 4.56],
        'S4': [61.13, 14.74, 0.72, 5.49, 2.99, 3.08, 1.91, 3.54, 0.09, 0.10, 6.23],
        'S5': [58.31, 14.33, 0.88, 6.47, 2.70, 3.98, 2.07, 3.13, 0.09, 0.12, 7.99],
        'S6': [60.28, 13.14, 0.82, 5.97, 2.57, 4.34, 2.05, 3.07, 0.10, 0.13, 7.38]
    })
    
    # Create a second depth with slightly modified values
    sample_data_2 = pd.DataFrame({
        'Elements': ['SiO2', 'Al2O3', 'TiO2', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'MnO', 'P2O5', 'LOI'],
        'S1': [62.45, 13.98, 0.71, 5.92, 3.05, 2.81, 1.98, 3.31, 0.08, 0.13, 5.58],
        'S2': [66.87, 12.45, 0.65, 4.78, 2.64, 2.83, 2.18, 2.89, 0.06, 0.12, 5.03],
        'S3': [67.42, 12.04, 0.67, 4.52, 2.55, 2.91, 2.03, 2.97, 0.07, 0.15, 4.67],
        'S4': [60.78, 15.02, 0.75, 5.61, 3.07, 3.14, 1.87, 3.58, 0.10, 0.11, 6.32],
        'S5': [57.96, 14.65, 0.92, 6.58, 2.76, 4.05, 2.03, 3.17, 0.10, 0.13, 8.15],
        'S6': [59.87, 13.41, 0.85, 6.09, 2.63, 4.42, 2.01, 3.12, 0.11, 0.14, 7.52]
    })
    
    # Create a third depth with different values
    sample_data_3 = pd.DataFrame({
        'Elements': ['SiO2', 'Al2O3', 'TiO2', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'MnO', 'P2O5', 'LOI'],
        'S1': [64.12, 12.97, 0.63, 5.45, 2.87, 2.62, 2.08, 3.21, 0.06, 0.11, 5.88],
        'S2': [68.34, 11.85, 0.60, 4.52, 2.48, 2.72, 2.26, 2.84, 0.04, 0.10, 4.97],
        'S3': [68.87, 11.54, 0.61, 4.28, 2.41, 2.79, 2.12, 2.89, 0.05, 0.13, 4.41],
        'S4': [62.05, 14.32, 0.68, 5.35, 2.91, 3.01, 1.96, 3.48, 0.08, 0.09, 6.07],
        'S5': [59.14, 13.98, 0.84, 6.31, 2.63, 3.89, 2.12, 3.07, 0.08, 0.11, 7.83],
        'S6': [61.25, 12.78, 0.78, 5.82, 2.49, 4.27, 2.10, 3.01, 0.09, 0.12, 7.21]
    })
    
    # Create an Excel file with three sheets
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sample_data_1.to_excel(writer, sheet_name='Depth_1', index=False)
        sample_data_2.to_excel(writer, sheet_name='Depth_2', index=False)
        sample_data_3.to_excel(writer, sheet_name='Depth_3', index=False)
    
    output.seek(0)
    return output
    
def main():
    st.set_page_config(
        page_title="TFA & CARTPAI Analysis", 
        page_icon=":bar_chart:", 
        layout="wide",
        initial_sidebar_state="collapsed"  # Start with sidebar collapsed for more space
    )
    
    # Add responsive CSS
    st.markdown("""
    <style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    .stApp {
        margin-top: 0 !important;
    }
    .block-container {
        padding-top: 0 !important;
    }
    /* Adjust spacing for better content flow */
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    /* Make file uploader more compact */
    .st-emotion-cache-1ol3fgf {
        padding: 1rem !important;
    }
    /* Adjust expander styling */
    .st-expander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    /* Responsive buttons */
    .stButton button {
        width: 100%;
    }
    @media (max-width: 768px) {
        .main-container {
            padding: 0.5rem;
        }
    }
    </style>
    <div class="main-container">
    """, unsafe_allow_html=True)
    
    # Add compact banner
    add_banner("./TFA Logo.png")
    
    st.write("""
    This application performs factor analysis on chemical composition data across multiple depths,
    and calculates class-level Paleo Affinity Index (PAI) values to assist in paleoenvironmental reconstruction.
    """)
    
    # Create columns for file upload and sample data options with responsive layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload with clearer instructions
        st.markdown("### Upload Your Data")
        uploaded_file = st.file_uploader(
            "Excel file with multiple sheets (one sheet per depth)",
            type=["xlsx"],
            help="Each sheet should contain chemical element data with elements in the first column"
        )
    
    with col2:
        # Sample data option with clearer styling
        st.markdown("### Try Sample Data")
        st.markdown(
            """
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <p style="margin: 0;">New to the tool? Try our sample dataset with three depths of geochemical data.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        sample_data = get_sample_data()
        
        # Download sample button
        st.download_button(
            label="Download Sample Data",
            data=sample_data,
            file_name="sample_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download sample data to see the expected format"
        )
        
        # Run with sample button
        use_sample = st.button(
            "Run Analysis with Sample Data", 
            help="Click to run the analysis using the sample dataset without uploading it"
        )
    
    # Process either uploaded file or sample data
    if uploaded_file is not None or use_sample:
        # Create a divider
        st.markdown('<hr style="margin: 1.5rem 0; border-color: #ddd;">', unsafe_allow_html=True)
        
        # Determine which data to use
        if use_sample:
            st.info("Using sample data for analysis")
            # Create a temporary file from the sample data
            sample_data_bytes = get_sample_data()
            file_to_analyze = sample_data_bytes
        else:
            file_to_analyze = uploaded_file
        
        try:
            # Read all sheets
            xl = pd.ExcelFile(file_to_analyze)
            sheet_names = xl.sheet_names
            
            if len(sheet_names) == 0:
                st.error("No sheets found in the Excel file.")
                return
            
            st.success(f"Found {len(sheet_names)} sheets (depths) in the file.")
            
            # Analysis parameters in a card-like container
            st.markdown(
                """
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="margin-top: 0;">Analysis Parameters</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Number of factors selection with better explanation
            num_factors = st.slider(
                "Number of factors for analysis", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="Select the number of factors to extract in the factor analysis"
            )
            
            # Create instance of ChemicalAnalysis
            analysis = ChemicalAnalysis()
            
            # Load data from each sheet
            dfs = []
            for sheet in sheet_names:
                df = pd.read_excel(file_to_analyze, sheet_name=sheet)
                dfs.append(df)
            
            # Process all depths with a more prominent button
            st.markdown('<div style="text-align: center; margin: 20px 0;">', unsafe_allow_html=True)
            analyze_button = st.button(
                "Analyze All Depths",
                key="analyze_button",
                help="Start the analysis process for all depths",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if analyze_button:
                with st.spinner("Analyzing all depths... This may take a moment."):
                    all_results = analysis.analyze_multiple_depths(dfs, num_factors)
                
                if not all_results:
                    st.error("Error processing data.")
                    return
                
                # Results section with tabs for better organization
                st.markdown('<div style="margin-top: 30px;">', unsafe_allow_html=True)
                tabs = st.tabs(["Results by Depth", "Comparative Analysis", "Download Results"])
                
                # Tab 1: Results by Depth
                with tabs[0]:
                    for result in all_results:
                        depth = result['depth']
                        
                        with st.expander(f"Depth {depth}", expanded=(depth == 1)):
                            # Create columns for better layout
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Display factor loadings
                                st.subheader(f"Factor Loadings - Depth {depth}")
                                st.dataframe(result['factor_analysis']['loadings'], use_container_width=True)
                            
                            with col2:
                                # Plot factor loadings
                                fig_loadings = analysis.plot_factor_loadings(result['factor_analysis']['loadings'], f"Depth {depth}")
                                st.pyplot(fig_loadings)
                            
                            # Display categorized scores
                            st.subheader(f"Categorized Factor Scores - Depth {depth}")
                            st.dataframe(result['categorized'], use_container_width=True)
                            
                            # Display Gini results
                            st.subheader(f"Gini Results - Depth {depth}")
                            for attr, (gini_df, _, weighted_gini) in result['pai_results']['gini_results'].items():
                                st.write(f"{attr} - Weighted Gini: {weighted_gini:.4f}")
                                st.dataframe(gini_df, use_container_width=True)
                            
                            # Display PAI results
                            st.subheader(f"Paleo Affinity Index (PAI) Results - Depth {depth}")
                            st.dataframe(result['pai_results']['pai_results'], use_container_width=True)
                            st.write(f"Total PAI: {result['pai_results']['total_pai']:.4f}")
                
                # Tab 2: Comparative Analysis
                with tabs[1]:
                    # Plot PAI comparison
                    st.subheader("Paleo Affinity Index (PAI) Values by Class Level Across Depths")
                    fig_pai = analysis.plot_pai_comparison(all_results)
                    st.plotly_chart(fig_pai, use_container_width=True)
                    
                    # Plot class level stability
                    st.subheader("Class Level PAI Stability")
                    fig_stability = analysis.plot_class_level_stability(all_results)
                    st.plotly_chart(fig_stability, use_container_width=True)
                
                # Tab 3: Download Results
                with tabs[2]:
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
                    
                    # Better styled download section
                    st.markdown(
                        """
                        <div style="background-color: #e9f7fe; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0;">
                            <h4>Download Complete Analysis Results</h4>
                            <p>The Excel file contains all analysis results with separate sheets for each depth.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.download_button(
                        label="Download All Results (Excel)",
                        data=output,
                        file_name="multi_depth_analysis_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Please check that your Excel file is properly formatted with one depth per sheet.")

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <h5 style="margin: 0; color: #4A5783;">TFA & CARTPAI Analysis Tool</h5>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #666;">Developed for paleoenvironmental reconstruction using Paleo Affinity Index (PAI)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Close main container div
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
