"""
Map VAE's 72 features to Model 2's 116 macro features
"""
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FeatureMapper:
    """Maps VAE output (72) to Model 2 input (116 macro features)"""
    
    def __init__(self, mapping_config: Dict):
        self.mapping = mapping_config
        
    def map_vae_to_model2(self, vae_output: Dict, latest_train_data: pd.Series) -> Dict:
        """
        Map VAE's 72 features to Model 2's 116 macro features
        
        Args:
            vae_output: Dict with 72 VAE-generated features
            latest_train_data: Latest row from training data (for company features only)
            
        Returns:
            Dict with ~116 macro features for Model 2
        """
        model2_features = {}
        
        # LOG what we're mapping
        logger.info(f"   Mapping VAE: GDP={vae_output.get('GDP', 0):.0f}, VIX={vae_output.get('VIX', 0):.2f}")
        
        # Direct mappings
        mappings = {
            "GDP": "GDP_last",
            "CPI": "CPI_last",
            "Unemployment_Rate": "Unemployment_Rate_last",
            "Federal_Funds_Rate": "Federal_Funds_Rate_mean",
            "VIX": "vix_q_mean",
            "SP500_Return_90D": "sp500_q_return",
            "Financial_Stress_Index": "Financial_Stress_Index_mean",
            "Corporate_Bond_Spread": "Corporate_Bond_Spread_mean",
            "High_Yield_Spread": "High_Yield_Spread_mean",
            "Oil_Price": "Oil_Price_mean",
            "TED_Spread": "TED_Spread_mean",
            "Treasury_10Y_Yield": "Treasury_10Y_Yield_mean",
            "Yield_Curve_Spread": "Yield_Curve_Spread_mean",
            "Consumer_Confidence": "Consumer_Confidence_mean",
            "Trade_Balance": "Trade_Balance_mean",
        }
        
        for vae_key, m2_key in mappings.items():
            if vae_key in vae_output:
                model2_features[m2_key] = float(vae_output[vae_key])
        
        # GDP lags
        if "GDP_Lag1" in vae_output:
            model2_features["GDP_last_lag_1q"] = vae_output["GDP_Lag1"]
        if "GDP_Lag5" in vae_output:
            model2_features["GDP_last_lag_2q"] = vae_output["GDP_Lag5"]
        if "GDP_Lag22" in vae_output:
            model2_features["GDP_last_lag_4q"] = vae_output["GDP_Lag22"]
        
        # Unemployment lags
        if "Unemployment_Rate_Lag1" in vae_output:
            model2_features["Unemployment_Rate_last_lag_1q"] = vae_output["Unemployment_Rate_Lag1"]
        if "Unemployment_Rate_Lag5" in vae_output:
            model2_features["Unemployment_Rate_last_lag_2q"] = vae_output["Unemployment_Rate_Lag5"]
        if "Unemployment_Rate_Lag22" in vae_output:
            model2_features["Unemployment_Rate_last_lag_4q"] = vae_output["Unemployment_Rate_Lag22"]
        
        # VIX features
        if "VIX" in vae_output:
            vix = vae_output["VIX"]
            model2_features["vix_q_mean"] = vix
            model2_features["vix_q_max"] = vix * 1.2
            model2_features["vix_q_std"] = abs(vix) * 0.3
            model2_features["vix_stress"] = 1.0 if vix > 30 else 0.0
        
        if "VIX_Lag1" in vae_output:
            model2_features["vix_q_mean_lag_1q"] = vae_output["VIX_Lag1"]
        if "VIX_Lag5" in vae_output:
            model2_features["vix_q_mean_lag_2q"] = vae_output["VIX_Lag5"]
        if "VIX_Lag22" in vae_output:
            model2_features["vix_q_mean_lag_4q"] = vae_output["VIX_Lag22"]
        
        if "VIX_MA90" in vae_output:
            model2_features["vix_q_mean_rolling4q_mean"] = vae_output["VIX_MA90"]
        if "VIX_Std22" in vae_output:
            model2_features["vix_q_mean_rolling4q_std"] = vae_output["VIX_Std22"]
        
        # SP500
        if "SP500_Close" in vae_output:
            sp500 = vae_output["SP500_Close"]
            model2_features["sp500_q_mean"] = sp500
            model2_features["sp500_q_start"] = sp500 * 0.98
            model2_features["sp500_q_end"] = sp500 * 1.02
            model2_features["sp500_q_max"] = sp500 * 1.05
            model2_features["sp500_q_min"] = sp500 * 0.95
        
        if "SP500_Return_90D" in vae_output:
            model2_features["sp500_q_return"] = vae_output["SP500_Return_90D"]
            model2_features["sp500_q_return_rolling4q_mean"] = vae_output["SP500_Return_90D"]
        
        if "SP500_Return_22D" in vae_output:
            model2_features["sp500_q_return_lag_1q"] = vae_output["SP500_Return_22D"]
        if "SP500_Return_5D" in vae_output:
            model2_features["sp500_q_return_lag_2q"] = vae_output["SP500_Return_5D"]
        
        if "SP500_Volatility_90D" in vae_output:
            model2_features["sp500_q_return_rolling4q_std"] = vae_output["SP500_Volatility_90D"]
        
        # Aggregates
        if "Federal_Funds_Rate" in vae_output:
            ffr = vae_output["Federal_Funds_Rate"]
            model2_features["Federal_Funds_Rate_max"] = ffr * 1.1
            model2_features["Federal_Funds_Rate_std"] = abs(ffr) * 0.2
        
        if "Financial_Stress_Index" in vae_output:
            fsi = vae_output["Financial_Stress_Index"]
            model2_features["Financial_Stress_Index_max"] = fsi * 1.3
            model2_features["Financial_Stress_Index_std"] = 0.5
        
        if "Corporate_Bond_Spread" in vae_output:
            cbs = vae_output["Corporate_Bond_Spread"]
            model2_features["Corporate_Bond_Spread_max"] = cbs * 1.2
            model2_features["Corporate_Bond_Spread_std"] = 0.3
        
        if "High_Yield_Spread" in vae_output:
            hys = vae_output["High_Yield_Spread"]
            model2_features["High_Yield_Spread_max"] = hys * 1.2
            model2_features["High_Yield_Spread_std"] = 0.4
        
        if "Consumer_Confidence" in vae_output:
            cc = vae_output["Consumer_Confidence"]
            model2_features["Consumer_Confidence_max"] = cc * 1.1
            model2_features["Consumer_Confidence_std"] = 5.0
        
        if "Oil_Price" in vae_output:
            oil = vae_output["Oil_Price"]
            model2_features["Oil_Price_max"] = oil * 1.15
            model2_features["Oil_Price_std"] = abs(oil) * 0.15
        
        if "Trade_Balance" in vae_output:
            tb = vae_output["Trade_Balance"]
            model2_features["Trade_Balance_max"] = tb * 1.1
            model2_features["Trade_Balance_std"] = 10
        
        if "Treasury_10Y_Yield" in vae_output:
            ty = vae_output["Treasury_10Y_Yield"]
            model2_features["Treasury_10Y_Yield_max"] = ty * 1.1
            model2_features["Treasury_10Y_Yield_std"] = 0.2
        
        if "Yield_Curve_Spread" in vae_output:
            ycs = vae_output["Yield_Curve_Spread"]
            model2_features["Yield_Curve_Spread_max"] = ycs * 1.2
            model2_features["Yield_Curve_Spread_std"] = 0.3
        
        if "TED_Spread" in vae_output:
            ted = vae_output["TED_Spread"]
            model2_features["TED_Spread_max"] = ted * 1.2
            model2_features["TED_Spread_std"] = 0.2
        
        # Yield curve inverted
        if "Yield_Curve_Inverted" in vae_output:
            model2_features["yield_curve_inverted"] = vae_output["Yield_Curve_Inverted"]
        
        # Unemployment stress
        unemp = vae_output.get("Unemployment_Rate", 5)
        model2_features["unemployment_stress"] = 1.0 if unemp > 8 else 0.0
        
        # Composite stress
        vix_val = vae_output.get("VIX", 20)
        gdp_growth = vae_output.get("GDP_Growth_252D", 0)
        stress = (vix_val/50 * 0.5) + (max(0, unemp-5)/10 * 0.3) + (max(0, -gdp_growth*20) * 0.2)
        model2_features["composite_stress_score"] = stress
        
        # Crisis flag
        model2_features["crisis_flag"] = 1.0 if (vix_val > 30 or unemp > 8) else 0.0
        
        # Company-specific features from training data (don't change with scenario)
        company_features = [
            "excess_return", "next_q_return", "pe_ratio", "return_calculated",
            "return_momentum", "return_vs_sector", "sector_avg_return",
            "q_price", "q_price_lag_1q", "q_price_lag_2q", "q_price_lag_4q",
            "q_price_range_pct", "q_return", "q_return_lag_1q", "q_return_lag_2q",
            "q_return_lag_4q", "q_return_rolling4q_max", "q_return_rolling4q_mean",
            "q_return_rolling4q_min", "q_return_rolling4q_std", "high_leverage",
            "leverage_x_vix", "liquidity_risk"
        ]
        
        for feat in company_features:
            if feat in latest_train_data.index and pd.notna(latest_train_data[feat]):
                try:
                    model2_features[feat] = float(latest_train_data[feat])
                except:
                    model2_features[feat] = 0.0
            else:
                model2_features[feat] = 0.0
        
        logger.info(f"   ✓ Final: {len(model2_features)} features mapped")
        
        return model2_features