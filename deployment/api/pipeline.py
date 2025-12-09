"""
Complete stress test pipeline: Model 1 → Model 2 → Model 3 with SHAP
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import shap

logger = logging.getLogger(__name__)

class StressTestPipeline:
    """End-to-end stress test pipeline with SHAP explanations"""
    
    def __init__(self, models: Dict, feature_mapper, data_fetcher, config):
        self.model1 = models["model1"]
        self.model2 = models["model2"]
        self.model3 = models["model3"]
        self.feature_mapper = feature_mapper
        self.data_fetcher = data_fetcher
        self.config = config
        self.scenarios = []
        
        # Initialize SHAP explainer for Model 3 (do once at startup)
        try:
            logger.info("🔍 Initializing SHAP explainer for Model 3...")
            # Use a small sample of training data as background
            sample_data = self._get_sample_background_data()
            self.shap_explainer = shap.KernelExplainer(
                self._model3_predict_wrapper,
                sample_data
            )
            logger.info("   ✓ SHAP explainer ready")
        except Exception as e:
            logger.warning(f"   ⚠️ SHAP initialization failed: {e}")
            self.shap_explainer = None
    
    def _get_sample_background_data(self, n_samples=10):
        """Get sample data for SHAP background (14 features only!)"""
        companies = list(self.data_fetcher.company_lookup.keys())[:n_samples]
        
        background_data = []
        for company_id in companies:
            company_data = self.data_fetcher.get_company_data(company_id)
            latest = company_data["latest"]
            
            # Create a sample with ONLY the 14 features Model 3 uses
            sample = {
                "GDP_last": 16000,
                "CPI_last": 250,
                "Unemployment_Rate_last": 5.0,
                "Federal_Funds_Rate_mean": 2.5,
                "vix_q_mean": 20,
                "sp500_q_return": 0.05,
                "Financial_Stress_Index_mean": 0,
                "Revenue": float(latest["Revenue"]),
                "Net_Income": float(latest["Net_Income"]),
                "Debt_to_Equity": float(latest["Debt_to_Equity"]),
                "Current_Ratio": float(latest.get("Current_Ratio", 1.0)),
                "net_margin": float(latest.get("net_margin", 0.1)),
                "roa": float(latest.get("roa", 0.05)),
                "roe": float(latest.get("roe", 0.10))
            }
            background_data.append(sample)
        
        # Return as numpy array with correct shape (n_samples, 14)
        df = pd.DataFrame(background_data)[self.config.MODEL3_FEATURES]
        
        logger.info(f"      Background data shape: {df.shape} (should be ({n_samples}, 14))")
        
        return df.values
    
    def _model3_predict_wrapper(self, X):
        """Wrapper for SHAP - returns risk scores"""
        # X is numpy array from SHAP (should be 14 features)
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # CRITICAL: X should already be 14 features from SHAP
        # Just scale and predict
        if self.model3["scaler"]:
            X_scaled = self.model3["scaler"].transform(X)
        else:
            X_scaled = X
        
        # Get decision function scores (negative = anomaly)
        scores = self.model3["model"].decision_function(X_scaled)
        
        # Convert to risk scores (0-100)
        risk_scores = []
        for score in scores:
            pred = -1 if score < 0 else 1
            
            # Use the same calibrated mapping
            if score < -5.0:
                risk = 95 + min(abs(score + 5.0), 5)
            elif score < -3.0:
                risk = 75 + (abs(score + 3.0) / 2.0) * 20
            elif score < -1.0:
                risk = 50 + (abs(score + 1.0) / 2.0) * 25
            elif score < 0.0:
                risk = 30 + abs(score) * 20
            elif score < 0.3:
                risk = 15 + (score / 0.3) * 15
            else:
                risk = max(0, 15 - (score - 0.3) * 20)
            
            risk_scores.append(risk)
        
        return np.array(risk_scores)
    
    def generate_scenarios(self, n_scenarios: int = 10) -> List[Dict]:
        """Generate scenarios using Model 1 (VAE)"""
        logger.info(f"🎲 Generating {n_scenarios} scenarios with Model 1...")
        
        scenarios = []
        
        for severity, params in self.config.SCENARIO_SEVERITIES.items():
            count = params["count"]
            sigma = params["sigma"]
            
            # Generate 'count' scenarios for this severity
            for i in range(count):
                if len(scenarios) >= n_scenarios:
                    break
                
                scenario_data = self._generate_single_scenario(sigma)
                
                scenarios.append({
                    "scenario_id": len(scenarios) + 1,
                    "severity": severity,
                    "sigma": sigma,
                    "crisis_type": self._determine_crisis_type(scenario_data, severity),
                    "features": scenario_data
                })
            
            if len(scenarios) >= n_scenarios:
                break
        
        self.scenarios = scenarios
        logger.info(f"   ✓ Generated {len(self.scenarios)} scenarios")
        
        return self.scenarios
    
    def _generate_single_scenario(self, sigma: float) -> Dict:
        """Generate single scenario using VAE scaler"""
        vae_data = self.model1
        scaler = vae_data.get("scaler")
        feature_names = vae_data.get("features", [])
        
        # Sample from normal distribution with specified sigma
        z_scaled = np.random.normal(0, sigma, size=len(feature_names))
        
        # Inverse transform to get realistic values
        generated_values = scaler.inverse_transform(z_scaled.reshape(1, -1))[0]
        
        # Create dict with feature names
        scenario_dict = {}
        for i, name in enumerate(feature_names):
            scenario_dict[name] = float(generated_values[i])
        
        return scenario_dict
    
    def _determine_crisis_type(self, features: Dict, severity: str) -> str:
        """Determine crisis type from features"""
        vix = features.get("VIX", 20)
        gdp = features.get("GDP", 16000)
        unemployment = features.get("Unemployment_Rate", 5)
        
        if severity == "baseline":
            return "Normal Economy"
        elif vix > 40:
            return "Market Panic"
        elif unemployment > 8:
            return "Recession"
        elif gdp < 14000:
            return "Economic Contraction"
        else:
            return "Moderate Stress"
    
    def run_stress_test(self, company_id: str, scenario_ids: List[int]) -> List[Dict]:
        """Run complete stress test with SHAP explanations"""
        logger.info(f"🔬 Running stress test: {company_id} × {len(scenario_ids)} scenarios")
        
        company_data = self.data_fetcher.get_company_data(company_id)
        latest_macro = self.data_fetcher.get_latest_macro_features()
        
        results = []
        
        for scenario_id in scenario_ids:
            scenario = next((s for s in self.scenarios if s["scenario_id"] == scenario_id), None)
            if not scenario:
                logger.warning(f"Scenario {scenario_id} not found")
                continue
            
            result = self._run_single_scenario(
                company_id=company_id,
                company_data=company_data,
                scenario=scenario,
                latest_macro=latest_macro
            )
            
            results.append(result)
        
        logger.info(f"   ✓ Completed {len(results)} stress tests")
        
        return results
    
    def _run_single_scenario(
        self, 
        company_id: str,
        company_data: Dict,
        scenario: Dict,
        latest_macro: pd.Series
    ) -> Dict:
        """Run pipeline for single company × scenario"""
        
        logger.info(f"\nScenario {scenario['scenario_id']} features:")
        vae_features = scenario["features"]
        logger.info(f"   VIX: {vae_features.get('VIX', 0):.2f}")
        logger.info(f"   GDP: {vae_features.get('GDP', 0):.0f}")
        logger.info(f"   Unemployment: {vae_features.get('Unemployment_Rate', 0):.2f}")
        
        # STEP 1: Map VAE features to Model 2 features
        model2_macro = self.feature_mapper.map_vae_to_model2(vae_features, latest_macro)
        
        # STEP 2: Prepare Model 2 input (211 features)
        latest = company_data["latest"]
        
        # Get all features (macro + company)
        all_features = model2_macro.copy()
        
        # Add company features (but don't overwrite macro!)
        exclude_cols = [
            'Date', 'Company', 'Company_Name', 'Sector', 'Year', 'Quarter',
            'Quarter_Num', 'Quarter_End_Date', 'Original_Quarter_End',
            'target_revenue', 'target_eps', 'target_debt_equity',
            'target_profit_margin', 'target_stock_return',
            'snorkel_label', 'prob_not_at_risk', 'prob_at_risk', 'AT_RISK',
            'crisis_flag', 'Unnamed: 0'
        ]
        
        for col in latest.index:
            if col not in exclude_cols:
                # CRITICAL: Don't overwrite mapped features!
                if col not in all_features:
                    all_features[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0
        
        logger.info(f"   Model 2 expects {self.model2['n_features']} features")
        
        # Get exact feature names from model
        feature_names = self.model2.get("feature_names")
        if not feature_names:
            logger.error("   ❌ No feature names found in model!")
            return self._create_error_result(company_id, scenario)
        
        # Create DataFrame with exact features in exact order
        X_model2 = pd.DataFrame([all_features])
        
        # Add one-hot encoded columns
        X_model2[f'Company_{company_id}'] = 1
        X_model2[f'Sector_{latest["Sector"]}'] = 1
        
        # Get only the features the model expects, in exact order
        final_features = {}
        for feat in feature_names:
            if feat in X_model2.columns:
                final_features[feat] = X_model2[feat].iloc[0]
            else:
                final_features[feat] = 0.0
        
        X_final = pd.DataFrame([final_features])[feature_names]
        
        logger.info(f"   ✓ Created {len(feature_names)} features in exact order")
        
        # STEP 3: Run Model 2 predictions (HANDLE DIFFERENT MODEL TYPES)
        logger.info(f"   → Running Model 2 predictions...")
        
        predictions = {}
        
        try:
            # Clean data before prediction
            X_array = X_final.values.astype('float64')
            
            # Validate data
            if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
                X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Clip extreme values
            X_array = np.clip(X_array, -1e10, 1e10)
            
            for target, model in self.model2["models"].items():
                model_type = type(model).__name__
                logger.info(f"      → {target} ({model_type})...")
                
                # Handle different LightGBM types
                if model_type == 'LGBMRegressor':
                    # Sklearn wrapper
                    pred = model.predict(X_array, num_threads=1)[0]
                    
                elif model_type == 'Booster':
                    # Raw LightGBM booster
                    import lightgbm as lgb
                    pred = model.predict(X_array, num_threads=1)[0]
                    
                else:
                    # Generic fallback
                    logger.warning(f"         Unknown model type: {model_type}")
                    pred = model.predict(X_array)[0]
                
                predictions[f"target_{target}"] = float(pred)
                logger.info(f"         ✓ {target}: {pred:.2f}")
            
            logger.info(f"   ✅ Model 2 predictions SUCCESSFUL!")
            
        except Exception as e:
            logger.error(f"   ❌ Model 2 failed: {e}")
            # Use fallback
            predictions = self._fallback_predictions(vae_features, latest)
        
        # STEP 4: Prepare Model 3 input (14 features) - USE CURRENT VALUES!
        model3_input = self._prepare_model3_input(
            vae_features=vae_features,
            predictions=predictions,
            company_current=latest
        )
        
        logger.info(f"   DEBUG Model 3 input (using CURRENT company values):")
        logger.info(f"      Revenue: {model3_input['Revenue']:.0f} (current)")
        logger.info(f"      Debt_to_Equity: {model3_input['Debt_to_Equity']:.2f} (current)")
        logger.info(f"      GDP_last: {model3_input['GDP_last']:.0f} (scenario)")
        logger.info(f"      vix_q_mean: {model3_input['vix_q_mean']:.2f} (scenario)")
        
        # STEP 5: Run Model 3 (anomaly detection) + SHAP
        X_model3 = pd.DataFrame([model3_input])[self.config.MODEL3_FEATURES]
        
        if self.model3["scaler"]:
            X_model3_scaled = self.model3["scaler"].transform(X_model3)
        else:
            X_model3_scaled = X_model3.values
        
        # Predict
        anomaly_score = self.model3["model"].decision_function(X_model3_scaled)[0]
        anomaly_pred = self.model3["model"].predict(X_model3_scaled)[0]
        
        # Convert to risk score
        risk_score = self._calculate_risk_score(anomaly_score, anomaly_pred)
        
        logger.info(f"   ✓ Risk score: {risk_score:.1f}")
        
        # STEP 6: Calculate SHAP values
        shap_explanations = self._calculate_shap_explanations(X_model3)
        
        # STEP 7: Format output
        return self._create_safe_output({
            "company_id": company_id,
            "scenario_id": scenario["scenario_id"],
            "scenario": {
                "severity": scenario["severity"],
                "crisis_type": scenario["crisis_type"],
                "sigma": float(scenario["sigma"]),
                "macroeconomic_indicators": {
                    "GDP": float(vae_features.get("GDP", 0)),
                    "GDP_growth_pct": float(vae_features.get("GDP_Growth_252D", 0)) * 100,
                    "CPI": float(vae_features.get("CPI", 0)),
                    "inflation_pct": float(vae_features.get("Inflation", 0)) * 100,
                    "unemployment_pct": float(vae_features.get("Unemployment_Rate", 0)),
                    "VIX": float(vae_features.get("VIX", 0)),
                    "SP500_close": float(vae_features.get("SP500_Close", 0)),
                    "SP500_return_pct": float(vae_features.get("SP500_Return_90D", 0)) * 100,
                    "oil_price": float(vae_features.get("Oil_Price", 0)),
                    "federal_funds_rate": float(vae_features.get("Federal_Funds_Rate", 0)),
                    "corporate_bond_spread": float(vae_features.get("Corporate_Bond_Spread", 0))
                }
            },
            "predictions": {
                "predicted_revenue": float(predictions["target_revenue"]),
                "predicted_eps": float(predictions["target_eps"]),
                "predicted_debt_equity": float(predictions["target_debt_equity"]),
                "predicted_profit_margin": float(predictions["target_profit_margin"]),
                "predicted_stock_return": float(predictions["target_stock_return"]),
                "current_revenue": float(latest["Revenue"]),
                "current_eps": float(latest["EPS"]),
                "current_debt_equity": float(latest["Debt_to_Equity"]),
                "current_profit_margin": float(latest.get("net_margin", 0)),
                "revenue_change_pct": float(((predictions["target_revenue"] - latest["Revenue"]) / latest["Revenue"] * 100) if latest["Revenue"] != 0 else 0),
                "eps_change_pct": float(((predictions["target_eps"] - latest["EPS"]) / latest["EPS"] * 100) if latest["EPS"] != 0 else 0),
                "debt_change_pct": float(((predictions["target_debt_equity"] - latest["Debt_to_Equity"]) / latest["Debt_to_Equity"] * 100) if latest["Debt_to_Equity"] != 0 else 0)
            },
            "risk_assessment": {
                "risk_score": float(risk_score),
                "risk_category": self._get_risk_category(risk_score),
                "anomaly_detected": bool(anomaly_pred == -1),
                "anomaly_score": float(anomaly_score),
                "anomaly_confidence": float(abs(anomaly_score)),
                "sector": str(latest["Sector"]),
                "model_used": "One-Class SVM",
                "model_roc_auc": 0.82,
                "interpretation": self._get_risk_interpretation(risk_score),
                "shap_explanations": shap_explanations
            }
        })
    
    def _calculate_shap_explanations(self, X_model3: pd.DataFrame) -> List[Dict]:
        """Calculate SHAP values for Model 3 predictions - SIMPLIFIED VERSION"""
        
        try:
            logger.info("   🔍 Calculating SHAP values...")
            
            # Use a faster Linear SHAP explainer for SVM
            # Create small background dataset (just 10 samples)
            background_samples = []
            companies = list(self.data_fetcher.company_lookup.keys())[:10]
            
            for company_id in companies:
                company_data = self.data_fetcher.get_company_data(company_id)
                latest = company_data["latest"]
                
                sample = {
                    "GDP_last": 16000, "CPI_last": 250, "Unemployment_Rate_last": 5.0,
                    "Federal_Funds_Rate_mean": 2.5, "vix_q_mean": 20, "sp500_q_return": 0.05,
                    "Financial_Stress_Index_mean": 0,
                    "Revenue": float(latest["Revenue"]),
                    "Net_Income": float(latest["Net_Income"]),
                    "Debt_to_Equity": float(latest["Debt_to_Equity"]),
                    "Current_Ratio": float(latest.get("Current_Ratio", 1.0)),
                    "net_margin": float(latest.get("net_margin", 0.1)),
                    "roa": float(latest.get("roa", 0.05)),
                    "roe": float(latest.get("roe", 0.10))
                }
                background_samples.append(sample)
            
            background_df = pd.DataFrame(background_samples)[self.config.MODEL3_FEATURES]
            
            # Create explainer with small sample size for speed
            explainer = shap.KernelExplainer(
                self._model3_predict_wrapper,
                background_df.values[:5],  # Use only 5 samples
                link="identity"
            )
            
            # Calculate SHAP values (use nsamples=50 for speed)
            shap_values = explainer.shap_values(X_model3.values, nsamples=50)
            
            # Get feature contributions
            explanations = []
            for i, feature in enumerate(self.config.MODEL3_FEATURES):
                shap_val = float(shap_values[0, i])
                feature_val = float(X_model3.iloc[0, i])
                
                explanations.append({
                    "feature": feature,
                    "feature_value": feature_val,
                    "shap_value": shap_val,
                    "contribution": self._get_contribution_description(shap_val),
                    "impact": "increases_risk" if shap_val > 0 else "decreases_risk"
                })
            
            # Sort by absolute SHAP value (most important first)
            explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            
            logger.info(f"   🔢 Normalizing SHAP values...")
            
            # NORMALIZE SHAP values to -100 to +100 scale for interpretability
            max_abs_shap = max(abs(e["shap_value"]) for e in explanations)
            logger.info(f"      Max absolute SHAP: {max_abs_shap:.2f}")
            
            if max_abs_shap > 0:
                for exp in explanations:
                    raw_value = exp["shap_value"]
                    normalized_value = (raw_value / max_abs_shap) * 100
                    
                    exp["shap_value_raw"] = raw_value  # Keep original
                    exp["shap_value_normalized"] = normalized_value
                    exp["shap_value"] = normalized_value  # Use normalized for display
                
                logger.info(f"      ✓ Normalized! Range now: [{min(e['shap_value'] for e in explanations):.1f}, {max(e['shap_value'] for e in explanations):.1f}]")
            
            logger.info(f"   ✓ SHAP calculated - Top 3 risk factors:")
            for exp in explanations[:3]:
                logger.info(f"      {exp['feature']}: {exp['shap_value']:+.1f} ({exp['contribution']})")
            
            return explanations
            
        except Exception as e:
            logger.warning(f"   ⚠️ SHAP calculation failed: {e}")
            logger.warning(f"   Using feature-based fallback explanations...")
            
            # FALLBACK: Simple feature-based explanations
            explanations = []
            for i, feature in enumerate(self.config.MODEL3_FEATURES):
                feature_val = float(X_model3.iloc[0, i])
                
                # Estimate contribution based on feature value
                if 'vix' in feature.lower():
                    impact = (feature_val - 20) * 2  # High VIX increases risk
                elif 'unemployment' in feature.lower():
                    impact = (feature_val - 5) * 3  # High unemployment increases risk
                elif 'gdp' in feature.lower():
                    impact = (16000 - feature_val) * 0.002  # Low GDP increases risk
                elif 'debt' in feature.lower():
                    impact = (feature_val - 1.5) * 5  # High debt increases risk
                elif 'margin' in feature.lower() or 'roa' in feature.lower() or 'roe' in feature.lower():
                    impact = (0.1 - feature_val) * 50  # Low profitability increases risk
                else:
                    impact = 0
                
                explanations.append({
                    "feature": feature,
                    "feature_value": feature_val,
                    "shap_value": impact,
                    "contribution": self._get_contribution_description(impact),
                    "impact": "increases_risk" if impact > 0 else "decreases_risk"
                })
            
            explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            logger.info(f"   ✓ Fallback explanations - Top 3 factors:")
            for exp in explanations[:3]:
                logger.info(f"      {exp['feature']}: {exp['shap_value']:+.3f}")
            
            return explanations
    
    def _get_contribution_description(self, shap_value: float) -> str:
        """Get human-readable description of SHAP contribution (normalized scale)"""
        abs_val = abs(shap_value)
        direction = "increases" if shap_value > 0 else "decreases"
        
        if abs_val > 50:
            return f"Strongly {direction} risk"
        elif abs_val > 25:
            return f"Moderately {direction} risk"
        elif abs_val > 10:
            return f"Slightly {direction} risk"
        else:
            return f"Minimal impact"
    
    def _prepare_model3_input(
        self,
        vae_features: Dict,
        predictions: Dict,
        company_current: pd.Series
    ) -> Dict:
        """
        Prepare 14 features for Model 3
        
        CRITICAL: Use CURRENT company values, NOT predictions!
        Model 3 was trained on actual current values.
        """
        
        return {
            # Macro features (7) - from VAE scenario
            "GDP_last": float(vae_features.get("GDP", 0)),
            "CPI_last": float(vae_features.get("CPI", 0)),
            "Unemployment_Rate_last": float(vae_features.get("Unemployment_Rate", 0)),
            "Federal_Funds_Rate_mean": float(vae_features.get("Federal_Funds_Rate", 0)),
            "vix_q_mean": float(vae_features.get("VIX", 0)),
            "sp500_q_return": float(vae_features.get("SP500_Return_90D", 0)),
            "Financial_Stress_Index_mean": float(vae_features.get("Financial_Stress_Index", 0)),
            
            # Company features (7) - USE CURRENT, NOT PREDICTIONS!
            "Revenue": float(company_current["Revenue"]),
            "Net_Income": float(company_current["Net_Income"]),
            "Debt_to_Equity": float(company_current["Debt_to_Equity"]),
            "Current_Ratio": float(company_current.get("Current_Ratio", 1.0)),
            "net_margin": float(company_current.get("net_margin", 0.1)),
            "roa": float(company_current.get("roa", 0.05)),
            "roe": float(company_current.get("roe", 0.10))
        }
    
    def _fallback_predictions(self, vae_features: Dict, latest: pd.Series) -> Dict:
        """Fallback stress-based predictions"""
        vix = vae_features.get("VIX", 20)
        unemployment = vae_features.get("Unemployment_Rate", 5)
        
        stress = min(1.0, (vix/50 + max(0, unemployment-5)/10) / 2)
        
        return {
            "target_revenue": float(latest["Revenue"]) * (1 - stress * 0.3),
            "target_eps": float(latest["EPS"]) * (1 - stress * 0.5),
            "target_debt_equity": float(latest["Debt_to_Equity"]) * (1 + stress * 0.2),
            "target_profit_margin": float(latest.get("net_margin", 0.1)) * (1 - stress * 0.3),
            "target_stock_return": -stress * 0.4
        }
    
    def _calculate_risk_score(self, anomaly_score: float, anomaly_pred: int) -> float:
        """
        Convert anomaly score to 0-100 risk score
        
        RECALIBRATED based on actual deployment scores:
        - Validation median: 0.152
        - Validation range: [-5.579, 0.805]
        - Real-world scenarios can be more extreme than training!
        
        New strategy: More realistic distribution
        """
        
        # Use wider ranges to prevent everything being CRITICAL
        
        if anomaly_score < -5.0:
            # Extremely rare (worse than worst in training)
            risk = 95 + min(abs(anomaly_score + 5.0), 5)  # 95-100
            
        elif anomaly_score < -3.0:
            # Very high risk (extreme scenarios)
            risk = 75 + (abs(anomaly_score + 3.0) / 2.0) * 20  # 75-95
            
        elif anomaly_score < -1.0:
            # High risk (stressed scenarios)
            risk = 50 + (abs(anomaly_score + 1.0) / 2.0) * 25  # 50-75
            
        elif anomaly_score < 0.0:
            # Moderate risk
            risk = 30 + abs(anomaly_score) * 20  # 30-50
            
        elif anomaly_score < 0.3:
            # Low risk (around median safe score)
            risk = 15 + (anomaly_score / 0.3) * 15  # 15-30
            
        else:
            # Very low risk (above 75th percentile)
            risk = max(0, 15 - (anomaly_score - 0.3) * 20)  # 0-15
        
        return round(float(risk), 1)
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Categorize risk score"""
        if risk_score < 25:
            return "LOW"
        elif risk_score < 50:
            return "MODERATE"
        elif risk_score < 75:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _get_risk_interpretation(self, risk_score: float) -> str:
        """Get interpretation of risk score"""
        if risk_score < 25:
            return "Company's current financial position shows strong resilience to this stress scenario"
        elif risk_score < 50:
            return "Company can likely withstand this stress with current financials, but monitoring advised"
        elif risk_score < 75:
            return "Significant vulnerability detected - company's current position faces elevated risk"
        else:
            return "Critical risk - company's current financial state is highly vulnerable to this scenario"
    
    def _create_safe_output(self, data: Dict) -> Dict:
        """Convert all numpy types to Python native types"""
        if isinstance(data, dict):
            return {k: self._create_safe_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._create_safe_output(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def _create_error_result(self, company_id: str, scenario: Dict) -> Dict:
        """Create error result"""
        return {
            "company_id": company_id,
            "scenario_id": scenario["scenario_id"],
            "error": "Pipeline execution failed",
            "risk_assessment": {
                "risk_score": 0.0,
                "risk_category": "UNKNOWN"
            }
        }