
# import pandas as pd
# import joblib
# import numpy as np
# import os
# import json
# from datetime import datetime
# from typing import Dict, Any, Optional, Tuple
# import warnings
# warnings.filterwarnings('ignore')

# class CompanyScenarioPredictionPipeline:
#     """
#     Robust pipeline for predicting company performance under economic scenarios
#     Uses 5 separate trained models for different target variables
#     """
    
#     def __init__(self, data_file_path: str, models_dir: str = "models/best_models", 
#                  output_dir: str = "prediction_results"):
#         """
#         Initialize the prediction pipeline
        
#         Args:
#             data_file_path: Path to quarterly_data_with_targets_clean.csv
#             models_dir: Directory containing the 5 .pkl model files
#             output_dir: Directory to save prediction results (JSON files)
#         """
#         self.data_file_path = data_file_path
#         self.models_dir = models_dir
#         self.output_dir = output_dir
        
#         # Create output directory if it doesn't exist
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         # Model file mapping
#         self.model_files = {
#             'debt_equity': 'debt_equity_best.pkl',
#             'eps': 'eps_best.pkl', 
#             'profit_margin': 'profit_margin_best.pkl',
#             'revenue': 'revenue_best.pkl',
#             'stock_return': 'stock_return_best.pkl'
#         }
        
#         # Initialize models dictionary
#         self.models = {}
        
#         # Load data and models
#         self._load_data()
#         self._load_models()
        
#         # Define scenario feature mapping
#         self._initialize_feature_mapping()
        
#         print(f"\nPipeline initialized successfully")
#         print(f"Data: {len(self.data):,} records loaded")
#         print(f"Companies: {len(self.data['Company'].unique())} unique companies")
#         print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
#         print(f"Models loaded: {len([m for m in self.models.values() if m is not None])}/{len(self.models)}\n")

#     def _load_data(self):
#         """Load the quarterly data file"""
#         try:
#             self.data = pd.read_csv(self.data_file_path)
            
#             # Ensure Date column is datetime
#             self.data['Date'] = pd.to_datetime(self.data['Date'])
            
#             # Sort by date for easier latest record retrieval
#             self.data = self.data.sort_values(['Company', 'Date'])
            
#         except Exception as e:
#             raise Exception(f"Error loading data file: {str(e)}")

#     def _load_models(self):
#         """Load all 5 trained models with intelligent extraction"""
        
#         for target_name, model_file in self.model_files.items():
#             model_path = os.path.join(self.models_dir, model_file)
            
#             try:
#                 loaded_object = joblib.load(model_path)
                
#                 # Check if it's directly a model
#                 if hasattr(loaded_object, 'predict'):
#                     self.models[target_name] = loaded_object
                    
#                 # Check if it's a dictionary containing a model
#                 elif isinstance(loaded_object, dict):
#                     model_found = False
                    
#                     # Common keys where models might be stored
#                     common_model_keys = ['model', 'best_model', 'estimator', 'best_estimator', 
#                                        'fitted_model', 'trained_model', target_name]
                    
#                     # First try common keys
#                     for key in common_model_keys:
#                         if key in loaded_object and hasattr(loaded_object[key], 'predict'):
#                             self.models[target_name] = loaded_object[key]
#                             model_found = True
#                             break
                    
#                     # If not found in common keys, search all keys
#                     if not model_found:
#                         for key, value in loaded_object.items():
#                             if hasattr(value, 'predict'):
#                                 self.models[target_name] = value
#                                 model_found = True
#                                 break
                    
#                     if not model_found:
#                         print(f"Warning: No model found in {model_file}")
#                         self.models[target_name] = None
                        
#                 else:
#                     print(f"Warning: Unknown object type in {model_file}")
#                     self.models[target_name] = None
                
#             except Exception as e:
#                 print(f"Error loading {model_file}: {str(e)}")
#                 self.models[target_name] = None

#     def _initialize_feature_mapping(self):
#         """Initialize comprehensive mapping from VAE features to training features"""
        
#         # Direct feature mappings (VAE → Predictor)
#         self.scenario_feature_mapping = {
#             # Core macro indicators
#             'GDP': 'GDP_last',
#             'CPI': 'CPI_last',
#             'Unemployment_Rate': 'Unemployment_Rate_last',
#             'Federal_Funds_Rate': 'Federal_Funds_Rate_mean',
#             'VIX': 'vix_q_mean',
#             'SP500_Close': 'sp500_q_mean',
            
#             # Economic indicators
#             'Oil_Price': 'Oil_Price_mean',
#             'Consumer_Confidence': 'Consumer_Confidence_mean',
#             'Yield_Curve_Spread': 'Yield_Curve_Spread_mean',
#             'TED_Spread': 'TED_Spread_mean',
#             'Treasury_10Y_Yield': 'Treasury_10Y_Yield_mean',
#             'Corporate_Bond_Spread': 'Corporate_Bond_Spread_mean',
#             'High_Yield_Spread': 'High_Yield_Spread_mean',
#             'Financial_Stress_Index': 'Financial_Stress_Index_mean',
#             'Trade_Balance': 'Trade_Balance_mean',
            
#             # Binary/categorical indicators
#             'Yield_Curve_Inverted': 'yield_curve_inverted',
#         }

#     def get_available_companies(self) -> list:
#         """Get list of available companies in the dataset"""
#         return sorted(self.data['Company'].unique().tolist())

#     def get_company_latest_data(self, company_name: str) -> Tuple[Dict[str, Any], str]:
#         """
#         Get the latest available data for a specific company
#         """
#         # Filter for the specific company
#         company_data = self.data[self.data['Company'] == company_name]
        
#         if company_data.empty:
#             available_companies = self.get_available_companies()
#             raise ValueError(f"Company '{company_name}' not found. Available companies: {available_companies[:10]}...")
        
#         # Get the latest record
#         latest_record = company_data.loc[company_data['Date'].idxmax()]
#         latest_date = latest_record['Date'].strftime('%Y-%m-%d')
        
#         # Convert to dictionary
#         company_dict = latest_record.to_dict()
        
#         return company_dict, latest_date

#     def apply_scenario_to_data(self, company_data: Dict[str, Any], 
#                               vae_scenario: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Apply VAE scenario to company data by replacing macro features
#         """
#         # Create a copy to avoid modifying original data
#         modified_data = company_data.copy()
        
#         changes_made = {}
        
#         # Apply direct mappings
#         for vae_feature, training_feature in self.scenario_feature_mapping.items():
#             if vae_feature in vae_scenario and training_feature in modified_data:
#                 original_value = modified_data[training_feature]
#                 new_value = vae_scenario[vae_feature]
#                 modified_data[training_feature] = new_value
#                 changes_made[training_feature] = (original_value, new_value)
        
#         # Apply derived features
#         self._apply_derived_features(modified_data, vae_scenario, changes_made)
        
#         # Remove target columns (these are what we're predicting)
#         target_columns = ['target_revenue', 'target_eps', 'target_debt_equity', 
#                          'target_profit_margin', 'target_stock_return']
#         for col in target_columns:
#             modified_data.pop(col, None)
        
#         return modified_data

#     def _apply_derived_features(self, modified_data: Dict[str, Any], 
#                                vae_scenario: Dict[str, Any], 
#                                changes_made: Dict[str, Tuple]) -> None:
#         """Apply derived features that need special handling"""
        
#         # Federal Funds Rate derivatives
#         if 'Federal_Funds_Rate' in vae_scenario:
#             rate = vae_scenario['Federal_Funds_Rate']
#             if 'Federal_Funds_Rate_max' in modified_data:
#                 modified_data['Federal_Funds_Rate_max'] = rate * 1.15
#             if 'Federal_Funds_Rate_std' in modified_data:
#                 modified_data['Federal_Funds_Rate_std'] = rate * 0.05
        
#         # VIX derivatives
#         if 'VIX' in vae_scenario:
#             vix = vae_scenario['VIX']
#             if 'vix_q_max' in modified_data:
#                 modified_data['vix_q_max'] = vix * 1.3
#             if 'vix_q_std' in modified_data:
#                 modified_data['vix_q_std'] = vix * 0.15
        
#         # SP500 derivatives
#         if 'SP500_Close' in vae_scenario:
#             sp500 = vae_scenario['SP500_Close']
#             if 'sp500_q_start' in modified_data:
#                 modified_data['sp500_q_start'] = sp500 * 0.97
#             if 'sp500_q_end' in modified_data:
#                 modified_data['sp500_q_end'] = sp500
#             if 'sp500_q_min' in modified_data:
#                 modified_data['sp500_q_min'] = sp500 * 0.92
#             if 'sp500_q_max' in modified_data:
#                 modified_data['sp500_q_max'] = sp500 * 1.08
        
#         # Stress indicators
#         vix_val = modified_data.get('vix_q_mean', 20)
#         unemployment_val = modified_data.get('Unemployment_Rate_last', 5)
        
#         if 'vix_stress' in modified_data:
#             modified_data['vix_stress'] = 1 if vix_val > 30 else 0
#         if 'unemployment_stress' in modified_data:
#             modified_data['unemployment_stress'] = 1 if unemployment_val > 7 else 0
#         if 'debt_accumulation' in modified_data:
#             stress_level = vae_scenario.get('Stress_Level', 'Low')
#             modified_data['debt_accumulation'] = 1 if stress_level == 'High' else 0
#         if 'yield_curve_inverted' in modified_data:
#             modified_data['yield_curve_inverted'] = int(vae_scenario.get('Yield_Curve_Inverted', 0))

#     def predict_all_targets(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Run predictions using all 5 models with feature alignment
#         """
#         predictions = {}
        
#         # Convert to DataFrame for model input
#         input_df = pd.DataFrame([prepared_data])
        
#         # Remove non-feature columns that might cause issues
#         columns_to_remove = ['Date', 'Company', 'Sector', 'Year', 'Quarter_Num', 'Quarter']
#         for col in columns_to_remove:
#             if col in input_df.columns:
#                 input_df = input_df.drop(columns=[col])
        
#         # Run each model with individual feature handling
#         for target_name, model in self.models.items():
#             if model is not None:
#                 try:
#                     # Create model-specific input
#                     model_input = input_df.copy()
                    
#                     # Handle feature alignment based on model requirements
#                     if hasattr(model, 'feature_names_in_'):
#                         expected_features = model.feature_names_in_
#                         model_input = model_input.reindex(columns=expected_features, fill_value=0)
                        
#                     elif hasattr(model, 'n_features_in_'):
#                         expected_count = model.n_features_in_
#                         current_count = len(model_input.columns)
                        
#                         if current_count < expected_count:
#                             # Add missing features with zeros
#                             for i in range(current_count, expected_count):
#                                 model_input[f'missing_feature_{i}'] = 0
#                         elif current_count > expected_count:
#                             # Take only first N features
#                             model_input = model_input.iloc[:, :expected_count]
                    
#                     # Special handling for LightGBM models
#                     model_type = str(type(model)).lower()
#                     if 'lightgbm' in model_type:
#                         try:
#                             import lightgbm as lgb
#                             if hasattr(model, 'predict'):
#                                 prediction = model.predict(model_input, predict_disable_shape_check=True)[0]
#                             else:
#                                 prediction = model.predict(model_input)[0]
#                         except Exception:
#                             predictions[target_name] = None
#                             continue
#                     else:
#                         # Standard prediction for non-LightGBM models
#                         prediction = model.predict(model_input)[0]
                    
#                     predictions[target_name] = float(prediction)
                    
#                 except Exception as e:
#                     predictions[target_name] = None
#             else:
#                 predictions[target_name] = None
        
#         return predictions

#     def save_results(self, results: Dict[str, Any], custom_filename: Optional[str] = None) -> str:
#         """
#         Save prediction results to a JSON file
        
#         Args:
#             results: The results dictionary from run_complete_prediction
#             custom_filename: Optional custom filename (without extension)
            
#         Returns:
#             Path to the saved file
#         """
#         try:
#             # Generate filename with timestamp
#             if custom_filename:
#                 filename = f"{custom_filename}.json"
#             else:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 company_name = results.get('company', 'unknown').replace(' ', '_')
#                 filename = f"{company_name}_{timestamp}.json"
            
#             filepath = os.path.join(self.output_dir, filename)
            
#             # Add metadata to results
#             results_with_metadata = results.copy()
#             results_with_metadata['saved_at'] = datetime.now().isoformat()
#             results_with_metadata['file_version'] = '1.0'
            
#             # Save to JSON with nice formatting
#             with open(filepath, 'w') as f:
#                 json.dump(results_with_metadata, f, indent=2, default=str)
            
#             return filepath
            
#         except Exception as e:
#             print(f"Error saving results: {str(e)}")
#             return None

#     def run_complete_prediction(self, company_name: str, 
#                               vae_scenario: Dict[str, Any],
#                               save_results: bool = True) -> Dict[str, Any]:
#         """
#         Complete end-to-end prediction pipeline
        
#         Args:
#             company_name: Name of the company to predict
#             vae_scenario: Dictionary with scenario parameters
#             save_results: Whether to save results to JSON file (default: True)
            
#         Returns:
#             Dictionary with prediction results
#         """
#         try:
#             # Print header
#             print("\n" + "="*70)
#             print(f"PREDICTION: {company_name}")
#             print("="*70)
            
#             # Step 1: Get company's latest data
#             company_data, latest_date = self.get_company_latest_data(company_name)
            
#             # Print company baseline info
#             print(f"\nBaseline Date:    {latest_date}")
#             print(f"Sector:           {company_data.get('Sector', 'N/A')}")
#             print(f"Latest Revenue:   ${company_data.get('Revenue_lag_1q', 0):,.0f}")
#             print(f"Latest Assets:    ${company_data.get('Total_Assets_lag_1q', 0):,.0f}")
            
#             # Print scenario info
#             print(f"\n{'─'*70}")
#             print(f"SCENARIO: {vae_scenario.get('Scenario', 'Custom Scenario')}")
#             print(f"{'─'*70}")
            
#             # Show key scenario parameters
#             key_params = ['GDP', 'Unemployment_Rate', 'VIX', 'SP500_Close', 'Federal_Funds_Rate']
#             for param in key_params:
#                 if param in vae_scenario:
#                     value = vae_scenario[param]
#                     if isinstance(value, (int, float)):
#                         print(f"  {param.replace('_', ' '):<25} {value:>15.2f}")
            
#             # Step 2: Apply scenario to data
#             prepared_data = self.apply_scenario_to_data(company_data, vae_scenario)
            
#             # Step 3: Run all predictions
#             predictions = self.predict_all_targets(prepared_data)
            
#             # Step 4: Format comprehensive results
#             results = {
#                 'company': company_name,
#                 'baseline_date': latest_date,
#                 'scenario': vae_scenario.get('Scenario', 'Custom Scenario'),
#                 'predictions': {
#                     'revenue': predictions.get('revenue'),
#                     'eps': predictions.get('eps'),
#                     'debt_to_equity': predictions.get('debt_equity'),
#                     'profit_margin': predictions.get('profit_margin'),
#                     'stock_return': predictions.get('stock_return')
#                 },
#                 'baseline_context': {
#                     'sector': company_data.get('Sector'),
#                     'last_revenue': company_data.get('Revenue_lag_1q'),
#                     'last_assets': company_data.get('Total_Assets_lag_1q'),
#                 },
#                 'status': 'success',
#                 'models_used': len([p for p in predictions.values() if p is not None])
#             }
            
#             # Print predictions
#             print(f"\n{'─'*70}")
#             print("PREDICTIONS")
#             print(f"{'─'*70}")
            
#             pred_labels = {
#                 'revenue': 'Revenue',
#                 'eps': 'Earnings Per Share (EPS)',
#                 'debt_to_equity': 'Debt-to-Equity Ratio',
#                 'profit_margin': 'Profit Margin (%)',
#                 'stock_return': 'Stock Return'
#             }
            
#             for key, label in pred_labels.items():
#                 value = results['predictions'].get(key)
#                 if value is not None:
#                     if key == 'revenue':
#                         print(f"  {label:<30} ${value:>15,.2f}")
#                     elif key == 'profit_margin':
#                         print(f"  {label:<30} {value:>15.2f}%")
#                     elif key == 'stock_return':
#                         print(f"  {label:<30} {value:>15.2%}")
#                     else:
#                         print(f"  {label:<30} {value:>15.4f}")
#                 else:
#                     print(f"  {label:<30} {'N/A':>15}")
            
#             print(f"\n{'─'*70}")
#             print(f"Models successful: {results['models_used']}/5")
            
#             # Save results if requested
#             if save_results:
#                 filepath = self.save_results(results)
#                 if filepath:
#                     print(f"Saved to: {filepath}")
            
#             print("="*70 + "\n")
            
#             return results
            
#         except Exception as e:
#             error_result = {
#                 'company': company_name,
#                 'scenario': vae_scenario.get('Scenario', 'Custom Scenario'),
#                 'predictions': None,
#                 'status': 'error',
#                 'error': str(e)
#             }
#             print(f"\nError: {str(e)}\n")
#             return error_result

#     def get_available_companies(self) -> list:
#         """Get list of available companies in the dataset"""
#         return sorted(self.data['Company'].unique().tolist())


# # =============================================================================
# # TEST FUNCTIONS
# # =============================================================================

# def test_pipeline():
#     """Test the complete pipeline"""
    
#     try:
#         print("\n" + "="*70)
#         print("TESTING PREDICTION PIPELINE")
#         print("="*70 + "\n")
        
#         # Step 1: Initialize pipeline (UPDATE THESE PATHS)
#         pipeline = CompanyScenarioPredictionPipeline(
#             data_file_path="data/features/quarterly_data_with_targets_clean.csv",  # UPDATE PATH
#             models_dir="models/best_models",  # UPDATE PATH
#             output_dir="prediction_results"  # Results will be saved here
#         )
        
#         # Step 2: Check what companies are available
#         companies = pipeline.get_available_companies()
#         print(f"Found {len(companies)} companies in dataset")
#         print(f"First 10: {', '.join(companies[:10])}\n")
        
#         if not companies:
#             print("ERROR: No companies found in dataset!")
#             return False
        
#         # Step 3: Create test scenario
#         test_scenario = {
#             'Scenario': 'Test Severe Recession',
#             'Severity': 0.9,
#             'Stress_Level': 'High',
#             'GDP': -2.5,
#             'CPI': 1.8,
#             'Unemployment_Rate': 8.5,
#             'Federal_Funds_Rate': 1.0,
#             'VIX': 45,
#             'SP500_Close': 3200,
#             'Oil_Price': 75,
#             'Consumer_Confidence': 65,
#             'Yield_Curve_Spread': -0.5,
#         }
        
#         # Step 4: Test with first available company
#         test_company = companies[0]
        
#         # Step 5: Run prediction
#         results = pipeline.run_complete_prediction(test_company, test_scenario)
        
#         # Step 6: Return success status
#         return results['status'] == 'success'
            
#     except Exception as e:
#         print(f"\nCRITICAL ERROR: {str(e)}")
#         print("\nTROUBLESHOOTING:")
#         print("1. Check file paths are correct")
#         print("2. Ensure CSV file exists and has correct format")
#         print("3. Ensure models directory exists with all 5 .pkl files\n")
#         return False


# if __name__ == "__main__":
#     success = test_pipeline()
    
#     if success:
#         print("="*70)
#         print("TEST SUCCESSFUL - Pipeline is ready for use!")
#         print("="*70 + "\n")
#     else:
#         print("="*70)
#         print("TEST FAILED - Please check troubleshooting steps")
#         print("="*70 + "\n")


import pandas as pd
import joblib
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class CompanyScenarioPredictionPipeline:
    """
    Robust pipeline for predicting company performance under economic scenarios
    Uses 5 separate trained models for different target variables
    
    FEATURE MAPPING APPROACH:
    - VAE generates 75 scenario features (economic indicators, lags, volatility, etc.)
    - Predictor models expect 157 features (scenario + company-specific data)
    - This pipeline maps all applicable VAE features to predictor features
    - Company-specific features (Revenue_lag_1q, Total_Assets, etc.) come from historical data
    - Derived features (_max, _std) are calculated from VAE base values
    """
    
    def __init__(self, data_file_path: str, models_dir: str = "models/best_models", 
                 output_dir: str = "prediction_results"):
        """
        Initialize the prediction pipeline
        
        Args:
            data_file_path: Path to quarterly_data_with_targets_clean.csv
            models_dir: Directory containing the 5 .pkl model files
            output_dir: Directory to save prediction results (JSON files)
        """
        self.data_file_path = data_file_path
        self.models_dir = models_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model file mapping
        self.model_files = {
            'debt_equity': 'debt_equity_best.pkl',
            'eps': 'eps_best.pkl', 
            'profit_margin': 'profit_margin_best.pkl',
            'revenue': 'revenue_best.pkl',
            'stock_return': 'stock_return_best.pkl'
        }
        
        # Initialize models dictionary
        self.models = {}
        
        # Load data and models
        self._load_data()
        self._load_models()
        
        # Define scenario feature mapping
        self._initialize_feature_mapping()
        
        print(f"\nPipeline initialized successfully")
        print(f"Data: {len(self.data):,} records loaded")
        print(f"Companies: {len(self.data['Company'].unique())} unique companies")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"Models loaded: {len([m for m in self.models.values() if m is not None])}/{len(self.models)}\n")

    def _load_data(self):
        """Load the quarterly data file"""
        try:
            self.data = pd.read_csv(self.data_file_path)
            
            # Ensure Date column is datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Sort by date for easier latest record retrieval
            self.data = self.data.sort_values(['Company', 'Date'])
            
        except Exception as e:
            raise Exception(f"Error loading data file: {str(e)}")

    def _load_models(self):
        """Load all 5 trained models with intelligent extraction"""
        
        for target_name, model_file in self.model_files.items():
            model_path = os.path.join(self.models_dir, model_file)
            
            try:
                loaded_object = joblib.load(model_path)
                
                # Check if it's directly a model
                if hasattr(loaded_object, 'predict'):
                    self.models[target_name] = loaded_object
                    
                # Check if it's a dictionary containing a model
                elif isinstance(loaded_object, dict):
                    model_found = False
                    
                    # Common keys where models might be stored
                    common_model_keys = ['model', 'best_model', 'estimator', 'best_estimator', 
                                       'fitted_model', 'trained_model', target_name]
                    
                    # First try common keys
                    for key in common_model_keys:
                        if key in loaded_object and hasattr(loaded_object[key], 'predict'):
                            self.models[target_name] = loaded_object[key]
                            model_found = True
                            break
                    
                    # If not found in common keys, search all keys
                    if not model_found:
                        for key, value in loaded_object.items():
                            if hasattr(value, 'predict'):
                                self.models[target_name] = value
                                model_found = True
                                break
                    
                    if not model_found:
                        print(f"Warning: No model found in {model_file}")
                        self.models[target_name] = None
                        
                else:
                    print(f"Warning: Unknown object type in {model_file}")
                    self.models[target_name] = None
                
            except Exception as e:
                print(f"Error loading {model_file}: {str(e)}")
                self.models[target_name] = None

    def _initialize_feature_mapping(self):
        """Initialize comprehensive mapping from VAE features to training features"""
        
        # Direct 1-to-1 mappings (VAE → Predictor)
        self.scenario_feature_mapping = {
            # === CURRENT VALUES (Base Economic Indicators) ===
            'GDP': 'GDP_last',
            'CPI': 'CPI_last',
            'Unemployment_Rate': 'Unemployment_Rate_last',
            'Federal_Funds_Rate': 'Federal_Funds_Rate_mean',
            'Yield_Curve_Spread': 'Yield_Curve_Spread_mean',
            'Consumer_Confidence': 'Consumer_Confidence_mean',
            'Oil_Price': 'Oil_Price_mean',
            'Trade_Balance': 'Trade_Balance_mean',
            'Corporate_Bond_Spread': 'Corporate_Bond_Spread_mean',
            'TED_Spread': 'TED_Spread_mean',
            'Treasury_10Y_Yield': 'Treasury_10Y_Yield_mean',
            'Financial_Stress_Index': 'Financial_Stress_Index_mean',
            'High_Yield_Spread': 'High_Yield_Spread_mean',
            
            # === VIX AND MARKET INDICATORS ===
            'VIX': 'vix_q_mean',
            'SP500_Close': 'sp500_q_end',  # Primary mapping to quarter-end value
            
            # === MOVING AVERAGES (map to mean/current) ===
            'Federal_Funds_Rate_MA30': 'Federal_Funds_Rate_mean',
            'Federal_Funds_Rate_MA90': 'Federal_Funds_Rate_mean',
            'Oil_Price_MA30': 'Oil_Price_mean',
            'Oil_Price_MA90': 'Oil_Price_mean',
            'VIX_MA5': 'vix_q_mean',
            'VIX_MA22': 'vix_q_mean',
            'VIX_MA90': 'vix_q_mean',
            'SP500_MA50': 'sp500_q_mean',
            'SP500_MA200': 'sp500_q_mean',
            
            # === VOLATILITY ===
            'VIX_Std22': 'vix_q_std',
            
            # === BINARY/CATEGORICAL ===
            'Yield_Curve_Inverted': 'yield_curve_inverted',
            
            # === QUARTERLY LAGS (use recent lags as proxies) ===
            # Map recent daily lags to 1-quarter-ago values
            'GDP_Lag1': 'GDP_last_lag_1q',
            'Unemployment_Rate_Lag1': 'Unemployment_Rate_last_lag_1q',
            'VIX_Lag1': 'vix_q_mean_lag_1q',
            
            # === RETURNS ===
            'SP500_Return_22D': 'sp500_q_return',  # 22 trading days ~ 1 month ~ 1/3 quarter
        }

    def get_available_companies(self) -> list:
        """Get list of available companies in the dataset"""
        return sorted(self.data['Company'].unique().tolist())

    def get_company_latest_data(self, company_name: str) -> Tuple[Dict[str, Any], str]:
        """
        Get the latest available data for a specific company
        """
        # Filter for the specific company
        company_data = self.data[self.data['Company'] == company_name]
        
        if company_data.empty:
            available_companies = self.get_available_companies()
            raise ValueError(f"Company '{company_name}' not found. Available companies: {available_companies[:10]}...")
        
        # Get the latest record
        latest_record = company_data.loc[company_data['Date'].idxmax()]
        latest_date = latest_record['Date'].strftime('%Y-%m-%d')
        
        # Convert to dictionary
        company_dict = latest_record.to_dict()
        
        return company_dict, latest_date

    def apply_scenario_to_data(self, company_data: Dict[str, Any], 
                              vae_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply VAE scenario to company data by replacing macro features
        """
        # Create a copy to avoid modifying original data
        modified_data = company_data.copy()
        
        changes_made = {}
        
        # Apply direct mappings
        for vae_feature, training_feature in self.scenario_feature_mapping.items():
            if vae_feature in vae_scenario and training_feature in modified_data:
                original_value = modified_data[training_feature]
                new_value = vae_scenario[vae_feature]
                modified_data[training_feature] = new_value
                changes_made[training_feature] = (original_value, new_value)
        
        # Apply derived features
        self._apply_derived_features(modified_data, vae_scenario, changes_made)
        
        # Count how many VAE features were actually used
        vae_features_in_scenario = len([k for k in vae_scenario.keys() if k not in ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score']])
        vae_features_mapped = len(changes_made)
        
        # Remove target columns (these are what we're predicting)
        target_columns = ['target_revenue', 'target_eps', 'target_debt_equity', 
                         'target_profit_margin', 'target_stock_return']
        for col in target_columns:
            modified_data.pop(col, None)
        
        return modified_data

    def _apply_derived_features(self, modified_data: Dict[str, Any], 
                               vae_scenario: Dict[str, Any], 
                               changes_made: Dict[str, Tuple]) -> None:
        """Apply derived features that need special handling"""
        
        # ========================================================================
        # FEDERAL FUNDS RATE DERIVATIVES
        # ========================================================================
        if 'Federal_Funds_Rate' in vae_scenario:
            rate = vae_scenario['Federal_Funds_Rate']
            if 'Federal_Funds_Rate_max' in modified_data:
                modified_data['Federal_Funds_Rate_max'] = rate * 1.10
            if 'Federal_Funds_Rate_std' in modified_data:
                modified_data['Federal_Funds_Rate_std'] = rate * 0.03
                
        # Also use MA values if provided
        if 'Federal_Funds_Rate_MA30' in vae_scenario:
            rate_ma = vae_scenario['Federal_Funds_Rate_MA30']
            if 'Federal_Funds_Rate_mean' in modified_data:
                modified_data['Federal_Funds_Rate_mean'] = rate_ma
                
        # Handle volatility from VAE
        if 'Federal_Funds_Rate_Volatility_30D' in vae_scenario:
            if 'Federal_Funds_Rate_std' in modified_data:
                modified_data['Federal_Funds_Rate_std'] = vae_scenario['Federal_Funds_Rate_Volatility_30D']
        
        # ========================================================================
        # VIX DERIVATIVES
        # ========================================================================
        if 'VIX' in vae_scenario:
            vix = vae_scenario['VIX']
            if 'vix_q_max' in modified_data:
                modified_data['vix_q_max'] = vix * 1.25
            if 'vix_q_std' in modified_data:
                # Use VIX_Std22 if available, otherwise estimate
                if 'VIX_Std22' in vae_scenario:
                    modified_data['vix_q_std'] = vae_scenario['VIX_Std22']
                else:
                    modified_data['vix_q_std'] = vix * 0.12
        
        # Handle VIX regime
        if 'VIX_Regime' in vae_scenario:
            vix_regime = vae_scenario['VIX_Regime']
            # VIX_Regime might be encoded (0=low, 1=medium, 2=high)
            # We can use this to adjust stress indicators
            
        # ========================================================================
        # SP500 DERIVATIVES
        # ========================================================================
        if 'SP500_Close' in vae_scenario:
            sp500 = vae_scenario['SP500_Close']
            
            # Set mean and end values
            if 'sp500_q_mean' in modified_data:
                modified_data['sp500_q_mean'] = sp500
            if 'sp500_q_end' in modified_data:
                modified_data['sp500_q_end'] = sp500
            
            # Set start/end values
            if 'sp500_q_start' in modified_data:
                # Assume quarter starts slightly lower
                modified_data['sp500_q_start'] = sp500 * 0.98
                
            # Set min/max for the quarter
            if 'sp500_q_min' in modified_data:
                # Use volatility if available
                if 'SP500_Volatility_22D' in vae_scenario:
                    vol = vae_scenario['SP500_Volatility_22D'] / 100  # Convert to decimal
                    modified_data['sp500_q_min'] = sp500 * (1 - vol * 2)
                    modified_data['sp500_q_max'] = sp500 * (1 + vol * 2)
                else:
                    modified_data['sp500_q_min'] = sp500 * 0.90
                    modified_data['sp500_q_max'] = sp500 * 1.10
                    
        # Handle SP500 returns
        if 'SP500_Return_22D' in vae_scenario:
            if 'sp500_q_return' in modified_data:
                modified_data['sp500_q_return'] = vae_scenario['SP500_Return_22D'] / 100  # Convert to decimal
        elif 'SP500_Return_90D' in vae_scenario:
            if 'sp500_q_return' in modified_data:
                modified_data['sp500_q_return'] = vae_scenario['SP500_Return_90D'] / 100
                
        # ========================================================================
        # OIL PRICE DERIVATIVES
        # ========================================================================
        if 'Oil_Price' in vae_scenario:
            oil = vae_scenario['Oil_Price']
            if 'Oil_Price_max' in modified_data:
                if 'Oil_Price_Volatility_30D' in vae_scenario:
                    vol = vae_scenario['Oil_Price_Volatility_30D'] / 100
                    modified_data['Oil_Price_max'] = oil * (1 + vol * 2)
                    modified_data['Oil_Price_std'] = oil * vol
                else:
                    modified_data['Oil_Price_max'] = oil * 1.15
                    modified_data['Oil_Price_std'] = oil * 0.08
                    
        # ========================================================================
        # OTHER ECONOMIC INDICATOR DERIVATIVES (max, std)
        # ========================================================================
        indicator_pairs = [
            ('Yield_Curve_Spread', 'Yield_Curve_Spread'),
            ('Consumer_Confidence', 'Consumer_Confidence'),
            ('Trade_Balance', 'Trade_Balance'),
            ('Corporate_Bond_Spread', 'Corporate_Bond_Spread'),
            ('TED_Spread', 'TED_Spread'),
            ('Treasury_10Y_Yield', 'Treasury_10Y_Yield'),
            ('Financial_Stress_Index', 'Financial_Stress_Index'),
            ('High_Yield_Spread', 'High_Yield_Spread'),
        ]
        
        for vae_name, pred_name in indicator_pairs:
            if vae_name in vae_scenario:
                value = vae_scenario[vae_name]
                # Set max (roughly 10% higher than mean)
                max_key = f'{pred_name}_max'
                if max_key in modified_data:
                    modified_data[max_key] = value * 1.10 if value > 0 else value * 0.90
                # Set std (roughly 5% of mean)
                std_key = f'{pred_name}_std'
                if std_key in modified_data:
                    modified_data[std_key] = abs(value * 0.05)
        
        # ========================================================================
        # UNEMPLOYMENT RATE DERIVATIVES
        # ========================================================================
        if 'Unemployment_Rate' in vae_scenario:
            unemp = vae_scenario['Unemployment_Rate']
            # Set lagged values if not already set
            if 'Unemployment_Rate_Lag1' in vae_scenario:
                if 'Unemployment_Rate_last_lag_1q' in modified_data:
                    modified_data['Unemployment_Rate_last_lag_1q'] = vae_scenario['Unemployment_Rate_Lag1']
                    
        # Handle unemployment volatility
        if 'Unemployment_Rate_Volatility_30D' in vae_scenario:
            # This could be used to adjust confidence in predictions or stress indicators
            pass
            
        # ========================================================================
        # GDP DERIVATIVES
        # ========================================================================
        if 'GDP' in vae_scenario:
            gdp = vae_scenario['GDP']
            # Set lagged values if provided
            if 'GDP_Lag1' in vae_scenario:
                if 'GDP_last_lag_1q' in modified_data:
                    modified_data['GDP_last_lag_1q'] = vae_scenario['GDP_Lag1']
                    
        # Handle GDP growth rates
        if 'GDP_Growth_90D' in vae_scenario:
            # This represents ~1 quarter of growth
            # Could be used to adjust GDP lag values
            pass
        if 'GDP_Growth_252D' in vae_scenario:
            # This represents ~1 year of growth
            pass
            
        # ========================================================================
        # CPI / INFLATION DERIVATIVES
        # ========================================================================
        if 'Inflation' in vae_scenario:
            # Inflation is calculated from CPI, might need special handling
            pass
        if 'Inflation_MA3M' in vae_scenario:
            # 3-month moving average of inflation
            pass
            
        # ========================================================================
        # STRESS INDICATORS (Binary features)
        # ========================================================================
        vix_val = modified_data.get('vix_q_mean', 20)
        unemployment_val = modified_data.get('Unemployment_Rate_last', 5)
        
        # VIX stress: VIX > 30 indicates high stress
        if 'vix_stress' in modified_data:
            if 'VIX' in vae_scenario:
                modified_data['vix_stress'] = 1 if vae_scenario['VIX'] > 30 else 0
            else:
                modified_data['vix_stress'] = 1 if vix_val > 30 else 0
                
        # Unemployment stress: Unemployment > 7% indicates stress
        if 'unemployment_stress' in modified_data:
            if 'Unemployment_Rate' in vae_scenario:
                modified_data['unemployment_stress'] = 1 if vae_scenario['Unemployment_Rate'] > 7 else 0
            else:
                modified_data['unemployment_stress'] = 1 if unemployment_val > 7 else 0
                
        # Debt accumulation: Based on stress level
        if 'debt_accumulation' in modified_data:
            stress_level = vae_scenario.get('Stress_Level', 'Low')
            if stress_level == 'High' or vae_scenario.get('Severity', 0) > 0.7:
                modified_data['debt_accumulation'] = 1
            else:
                modified_data['debt_accumulation'] = 0
                
        # Yield curve inverted
        if 'yield_curve_inverted' in modified_data:
            if 'Yield_Curve_Inverted' in vae_scenario:
                modified_data['yield_curve_inverted'] = int(vae_scenario['Yield_Curve_Inverted'])
            elif 'Yield_Curve_Spread' in vae_scenario:
                # If spread is negative, curve is inverted
                modified_data['yield_curve_inverted'] = 1 if vae_scenario['Yield_Curve_Spread'] < 0 else 0
                
        # TED Spread High indicator
        if 'TED_Spread_High' in vae_scenario:
            # This might be a binary indicator (1 if TED spread is unusually high)
            # Could map to a stress indicator
            pass

    def predict_all_targets(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run predictions using all 5 models with feature alignment
        """
        predictions = {}
        
        # Convert to DataFrame for model input
        input_df = pd.DataFrame([prepared_data])
        
        # Remove non-feature columns that might cause issues
        columns_to_remove = ['Date', 'Company', 'Sector', 'Year', 'Quarter_Num', 'Quarter']
        for col in columns_to_remove:
            if col in input_df.columns:
                input_df = input_df.drop(columns=[col])
        
        # Run each model with individual feature handling
        for target_name, model in self.models.items():
            if model is not None:
                try:
                    # Create model-specific input
                    model_input = input_df.copy()
                    
                    # Handle feature alignment based on model requirements
                    if hasattr(model, 'feature_names_in_'):
                        expected_features = model.feature_names_in_
                        model_input = model_input.reindex(columns=expected_features, fill_value=0)
                        
                    elif hasattr(model, 'n_features_in_'):
                        expected_count = model.n_features_in_
                        current_count = len(model_input.columns)
                        
                        if current_count < expected_count:
                            # Add missing features with zeros
                            for i in range(current_count, expected_count):
                                model_input[f'missing_feature_{i}'] = 0
                        elif current_count > expected_count:
                            # Take only first N features
                            model_input = model_input.iloc[:, :expected_count]
                    
                    # Special handling for LightGBM models
                    model_type = str(type(model)).lower()
                    if 'lightgbm' in model_type:
                        try:
                            import lightgbm as lgb
                            if hasattr(model, 'predict'):
                                prediction = model.predict(model_input, predict_disable_shape_check=True)[0]
                            else:
                                prediction = model.predict(model_input)[0]
                        except Exception:
                            predictions[target_name] = None
                            continue
                    else:
                        # Standard prediction for non-LightGBM models
                        prediction = model.predict(model_input)[0]
                    
                    predictions[target_name] = float(prediction)
                    
                except Exception as e:
                    predictions[target_name] = None
            else:
                predictions[target_name] = None
        
        return predictions

    def save_results(self, results: Dict[str, Any], custom_filename: Optional[str] = None) -> str:
        """
        Save prediction results to a JSON file
        
        Args:
            results: The results dictionary from run_complete_prediction
            custom_filename: Optional custom filename (without extension)
            
        Returns:
            Path to the saved file
        """
        try:
            # Generate filename with timestamp
            if custom_filename:
                filename = f"{custom_filename}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                company_name = results.get('company', 'unknown').replace(' ', '_')
                filename = f"{company_name}_{timestamp}.json"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Add metadata to results
            results_with_metadata = results.copy()
            results_with_metadata['saved_at'] = datetime.now().isoformat()
            results_with_metadata['file_version'] = '1.0'
            
            # Save to JSON with nice formatting
            with open(filepath, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            
            return filepath
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return None

    def run_complete_prediction(self, company_name: str, 
                              vae_scenario: Dict[str, Any],
                              save_results: bool = True) -> Dict[str, Any]:
        """
        Complete end-to-end prediction pipeline
        
        Args:
            company_name: Name of the company to predict
            vae_scenario: Dictionary with scenario parameters
            save_results: Whether to save results to JSON file (default: True)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Print header
            print("\n" + "="*70)
            print(f"PREDICTION: {company_name}")
            print("="*70)
            
            # Step 1: Get company's latest data
            company_data, latest_date = self.get_company_latest_data(company_name)
            
            # Print company baseline info
            print(f"\nBaseline Date:    {latest_date}")
            print(f"Sector:           {company_data.get('Sector', 'N/A')}")
            print(f"Latest Revenue:   ${company_data.get('Revenue_lag_1q', 0):,.0f}")
            print(f"Latest Assets:    ${company_data.get('Total_Assets_lag_1q', 0):,.0f}")
            
            # Print scenario info
            print(f"\n{'─'*70}")
            print(f"SCENARIO: {vae_scenario.get('Scenario', 'Custom Scenario')}")
            print(f"{'─'*70}")
            
            # Show key scenario parameters
            key_params = ['GDP', 'Unemployment_Rate', 'VIX', 'SP500_Close', 'Federal_Funds_Rate']
            for param in key_params:
                if param in vae_scenario:
                    value = vae_scenario[param]
                    if isinstance(value, (int, float)):
                        print(f"  {param.replace('_', ' '):<25} {value:>15.2f}")
            
            # Step 2: Apply scenario to data
            prepared_data = self.apply_scenario_to_data(company_data, vae_scenario)
            
            # Count VAE features used
            vae_features_count = len([k for k in vae_scenario.keys() if k not in ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score']])
            print(f"\nVAE Features: {vae_features_count} provided from scenario")
            
            # Step 3: Run all predictions
            predictions = self.predict_all_targets(prepared_data)
            
            # Step 4: Format comprehensive results
            results = {
                'company': company_name,
                'baseline_date': latest_date,
                'scenario': vae_scenario.get('Scenario', 'Custom Scenario'),
                'predictions': {
                    'revenue': predictions.get('revenue'),
                    'eps': predictions.get('eps'),
                    'debt_to_equity': predictions.get('debt_equity'),
                    'profit_margin': predictions.get('profit_margin'),
                    'stock_return': predictions.get('stock_return')
                },
                'baseline_context': {
                    'sector': company_data.get('Sector'),
                    'last_revenue': company_data.get('Revenue_lag_1q'),
                    'last_assets': company_data.get('Total_Assets_lag_1q'),
                },
                'status': 'success',
                'models_used': len([p for p in predictions.values() if p is not None])
            }
            
            # Print predictions
            print(f"\n{'─'*70}")
            print("PREDICTIONS")
            print(f"{'─'*70}")
            
            pred_labels = {
                'revenue': 'Revenue',
                'eps': 'Earnings Per Share (EPS)',
                'debt_to_equity': 'Debt-to-Equity Ratio',
                'profit_margin': 'Profit Margin (%)',
                'stock_return': 'Stock Return'
            }
            
            for key, label in pred_labels.items():
                value = results['predictions'].get(key)
                if value is not None:
                    if key == 'revenue':
                        print(f"  {label:<30} ${value:>15,.2f}")
                    elif key == 'profit_margin':
                        print(f"  {label:<30} {value:>15.2f}%")
                    elif key == 'stock_return':
                        print(f"  {label:<30} {value:>15.2%}")
                    else:
                        print(f"  {label:<30} {value:>15.4f}")
                else:
                    print(f"  {label:<30} {'N/A':>15}")
            
            print(f"\n{'─'*70}")
            print(f"Models successful: {results['models_used']}/5")
            
            # Save results if requested
            if save_results:
                filepath = self.save_results(results)
                if filepath:
                    print(f"Saved to: {filepath}")
            
            print("="*70 + "\n")
            
            return results
            
        except Exception as e:
            error_result = {
                'company': company_name,
                'scenario': vae_scenario.get('Scenario', 'Custom Scenario'),
                'predictions': None,
                'status': 'error',
                'error': str(e)
            }
            print(f"\nError: {str(e)}\n")
            return error_result


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_pipeline():
    """Test the complete pipeline"""
    
    try:
        print("\n" + "="*70)
        print("TESTING PREDICTION PIPELINE")
        print("="*70 + "\n")
        
        # Step 1: Initialize pipeline (UPDATE THESE PATHS)
        pipeline = CompanyScenarioPredictionPipeline(
            data_file_path="data/features/quarterly_data_with_targets_clean.csv",  # UPDATE PATH
            models_dir="models/best_models",  # UPDATE PATH
            output_dir="prediction_results"  # Results will be saved here
        )
        
        # Step 2: Check what companies are available
        companies = pipeline.get_available_companies()
        print(f"Found {len(companies)} companies in dataset")
        print(f"First 10: {', '.join(companies[:10])}\n")
        
        if not companies:
            print("ERROR: No companies found in dataset!")
            return False
        
        # Step 3: Create test scenario with FULL VAE features
        # test_scenario = {
        #     'Scenario': 'Test Severe Recession',
        #     'Severity': 0.9,
        #     'Stress_Level': 'High',
        #     'GDP': -2.5,
        #     'CPI': 250.0,
        #     'Unemployment_Rate': 8.5,
        #     'Federal_Funds_Rate': 1.0,
        #     'VIX': 45,
        #     'SP500_Close': 3200,
        #     'Oil_Price': 75,
        #     'Consumer_Confidence': 65,
        #     'Yield_Curve_Spread': -0.5,
        #     'Trade_Balance': -50000,
        #     'Corporate_Bond_Spread': 2.5,
        #     'TED_Spread': 0.5,
        #     'Treasury_10Y_Yield': 2.0,
        #     'Financial_Stress_Index': 1.2,
        #     'High_Yield_Spread': 6.5,
        #     # Add lag features
        #     'GDP_Lag1': -1.5,
        #     'Unemployment_Rate_Lag1': 7.0,
        #     'VIX_Lag1': 35,
        #     # Add volatility features
        #     'VIX_Std22': 5.5,
        #     'SP500_Volatility_22D': 25,
        #     'Oil_Price_Volatility_30D': 15,
        #     # Add returns
        #     'SP500_Return_22D': -8.5,
        #     'Yield_Curve_Inverted': 1,
        # }


        # Step 3: Create test scenario - REPLACE THIS ENTIRE BLOCK
        test_scenario = {
            'Scenario': 'Scenario_5_Baseline_Mild_Stress',
            'Severity': 3,
            'Stress_Level': 'Mild Stress',
            'Stress_Score': 3,
            
            # Base economic indicators
            'GDP': 17221.473,
            'CPI': 215.40843,
            'Unemployment_Rate': 5.313863,
            'Federal_Funds_Rate': 0.7972529,
            'Yield_Curve_Spread': 2.5636744,
            'Consumer_Confidence': 93.709625,
            'Oil_Price': 34.888336,
            'Trade_Balance': -35332.93,
            'Corporate_Bond_Spread': 2.7262948,
            'TED_Spread': 0.30649337,
            'Treasury_10Y_Yield': 3.3017654,
            'Financial_Stress_Index': 0.13574266,
            'High_Yield_Spread': 5.993139,
            
            # GDP lags
            'GDP_Lag1': 17220.139,
            'GDP_Lag5': 17213.66,
            'GDP_Lag22': 17194.076,
            
            # CPI lags
            'CPI_Lag1': 215.37982,
            'CPI_Lag5': 215.26305,
            'CPI_Lag22': 214.94107,
            
            # Unemployment lags
            'Unemployment_Rate_Lag1': 5.3149514,
            'Unemployment_Rate_Lag5': 5.314064,
            'Unemployment_Rate_Lag22': 5.328718,
            
            # Federal Funds Rate lags
            'Federal_Funds_Rate_Lag1': 0.7969537,
            'Federal_Funds_Rate_Lag5': 0.79424906,
            'Federal_Funds_Rate_Lag22': 0.79364437,
            
            # Yield Curve Spread lags
            'Yield_Curve_Spread_Lag1': 2.5548043,
            'Yield_Curve_Spread_Lag5': 2.5358906,
            'Yield_Curve_Spread_Lag22': 2.5232499,
            
            # Oil Price lags
            'Oil_Price_Lag1': 34.610607,
            'Oil_Price_Lag5': 34.05779,
            'Oil_Price_Lag22': 32.187614,
            
            # Consumer Confidence lags
            'Consumer_Confidence_Lag1': 93.62604,
            'Consumer_Confidence_Lag5': 93.39721,
            'Consumer_Confidence_Lag22': 92.99233,
            
            # Growth and inflation
            'GDP_Growth_90D': 0.8596115,
            'GDP_Growth_252D': 1.8838522,
            'Inflation': 0.14552055,
            'Inflation_MA3M': 0.0652416,
            
            # Moving averages
            'Unemployment_Rate_MA30': 5.3222723,
            'Unemployment_Rate_MA90': 5.281821,
            'Federal_Funds_Rate_MA30': 0.7925731,
            'Federal_Funds_Rate_MA90': 0.8260282,
            'Oil_Price_MA30': 33.07557,
            'Oil_Price_MA90': 31.343412,
            
            # Volatility
            'Oil_Price_Volatility_30D': 1.6464845,
            'Unemployment_Rate_Volatility_30D': 0.044576496,
            'Federal_Funds_Rate_Volatility_30D': 0.008272618,
            
            # Binary indicators
            'Yield_Curve_Inverted': 0,
            'TED_Spread_High': 0.040310334,
            
            # VIX and market
            'VIX': 15.203285,
            'SP500_Close': 1718.6337,
            
            # VIX lags
            'VIX_Lag1': 15.641421,
            'VIX_Lag5': 16.20053,
            'VIX_Lag22': 17.27496,
            
            # VIX moving averages
            'VIX_MA5': 15.766788,
            'VIX_MA22': 16.45789,
            'VIX_MA90': 18.967962,
            
            # VIX statistics
            'VIX_Std22': 1.1477562,
            'VIX_Regime': 1.5862422,
            
            # SP500 returns
            'SP500_Return_1D': 0.4601572,
            'SP500_Return_5D': 1.306865,
            'SP500_Return_22D': 2.6190023,
            'SP500_Return_90D': 4.969393,
            
            # SP500 moving averages
            'SP500_MA50': 1674.3269,
            'SP500_MA200': 1648.4554,
            
            # SP500 vs moving averages
            'SP500_vs_MA50': 1.0267808,
            'SP500_vs_MA200': 1.0449194,
            
            # SP500 volatility
            'SP500_Volatility_22D': 11.6727295,
            'SP500_Volatility_90D': 14.537488,
            
            # SP500 momentum
            'SP500_Momentum_22D': 2.6189957,
            'SP500_Momentum_90D': 4.9693956,
            
            # Technical indicators
            'SP500_RSI_14D': 64.42662,
        }
        
        # Step 4: Test with first available company
        test_company = companies[0]
        
        # Step 5: Run prediction
        results = pipeline.run_complete_prediction(test_company, test_scenario)
        
        # Step 6: Return success status
        return results['status'] == 'success'
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Check file paths are correct")
        print("2. Ensure CSV file exists and has correct format")
        print("3. Ensure models directory exists with all 5 .pkl files\n")
        return False


if __name__ == "__main__":
    success = test_pipeline()
    
    if success:
        print("="*70)
        print("TEST SUCCESSFUL - Pipeline is ready for use!")
        print("="*70 + "\n")
    else:
        print("="*70)
        print("TEST FAILED - Please check troubleshooting steps")
        print("="*70 + "\n")