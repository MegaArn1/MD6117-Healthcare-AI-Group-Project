# Figure inventory

| Location | Figures | Source |
|---|---|---|
| [notebook/](notebook/) | Five separately exported EDA, ROC/PR, calibration and feature-count plots | Default CPU replay of `In_Hospital_Death.ipynb` |
| [L4_shap_summary.png](L4_shap_summary.png) | Global SHAP beeswarm | Retained historical XGBoost explanation |
| [L4_shap_waterfall_TP_high_risk_died.png](L4_shap_waterfall_TP_high_risk_died.png) | Correct high-risk prediction | Retained historical case |
| [L4_shap_waterfall_FP_high_risk_survived.png](L4_shap_waterfall_FP_high_risk_survived.png) | False-positive case | Retained historical case |
| [L4_shap_waterfall_FN_low_risk_died.png](L4_shap_waterfall_FN_low_risk_died.png) | False-negative case | Retained historical case |
| [L4_calibration_curve.png](L4_calibration_curve.png) | Original reliability curve | Retained historical test calibration |

The [notebook output index](../notebook/README.md) provides previews, intended use, source populations and links to the underlying tables. All PNGs are directly accessible without extracting images from the notebook. The two calibration PNGs visualise the same frozen model/data; they are not separate experiments.
