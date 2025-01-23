import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from an Excel file."""
    return pd.read_excel(file_path)

def preprocess_data(df):
    """Preprocess the data by scaling and performing statistical tests."""
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    chi_square_value, p_value = calculate_bartlett_sphericity(scaled_data)
    print(f"Bartlett's test: Chi-square value = {chi_square_value}, p-value = {p_value:.2f}")
    
    kmo_all, kmo_model = calculate_kmo(scaled_data)
    print(f"KMO test: KMO value = {kmo_model:.2f}")
    
    return scaled_data

def perform_factor_analysis(scaled_data, factor_n):
    """Perform exploratory factor analysis."""
    fa = FactorAnalyzer(n_factors=factor_n, rotation="varimax")
    fa.fit(scaled_data)
    
    loadings = pd.DataFrame(fa.loadings_, columns=[f'Factor {i+1}' for i in range(factor_n)], index=scaled_data.columns)
    loadings['communalities'] = fa.get_communalities()
    loadings['eigenvalues'] = fa.get_eigenvalues()
    
    return loadings

def plot_scree_plot(scaled_data):
    """Generate and display a scree plot."""
    fa = FactorAnalyzer(n_factors=scaled_data.shape[1], rotation=None)
    fa.fit(scaled_data)
    
    eigenvalues = fa.get_eigenvalues()[0]
    plt.figure(figsize=(8, 5))
    plt.scatter(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()