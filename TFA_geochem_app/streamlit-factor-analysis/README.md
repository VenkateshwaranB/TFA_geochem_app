# README.md

# Streamlit Factor Analysis Application

This project is a Streamlit application designed to perform factor analysis and decision tree calculations. It provides an interactive interface for users to input data, visualize results, and understand the relationships between different factors.

## Project Structure

```
streamlit-factor-analysis
├── src
│   ├── app.py
│   ├── components
│   │   ├── factor_analysis.py
│   │   └── decision_tree.py
│   ├── utils
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   └── models
│       └── decision_tree_pai.py
├── data
│   └── sample_data.xlsx
├── tests
│   ├── __init__.py
│   ├── test_factor_analysis.py
│   └── test_decision_tree.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-factor-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the Streamlit application, run the following command in your terminal:

```
streamlit run src/app.py
```

Once the application is running, you can interact with the interface to perform factor analysis and decision tree calculations using the provided sample data.

## Testing

To run the unit tests for this application, navigate to the `tests` directory and execute:

```
pytest
```

This will run all the tests defined in `test_factor_analysis.py` and `test_decision_tree.py`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.