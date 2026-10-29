# Population-Level Public Health Surveillance: CDC Dataset 2 Epidemiological Analysis

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Secondary Dataset:** CDC United States Diabetes Surveillance System (USDSS) / National Health Interview Survey (NHIS)  
**Temporal Scope:** 2000 to 2024 (25 annual surveillance cycles)  
**Raw Data File:** `data/raw/export.csv` / `data/processed/df2_surveillance.csv` ($N = 13,889$ aggregate surveillance records)  

---

## 1. Dataset Documentation and Data Schema

Dataset 2 provides aggregate epidemiological surveillance records tracking longitudinal diabetes prevalence across all 50 U.S. states, the District of Columbia, and national benchmarks:

| Column Name | Data Type | Example Value | Description |
|:---|:---:|:---:|:---|
| `Year` | Integer | 2024 | Survey reporting year (2000 to 2024) |
| `State` | String | `United States` / `Georgia` | State or national aggregate territory |
| `Age` | String | `Crude` / `18-44` / `45-64` / `65+` | Age stratification category |
| `Population` | String | `All Ages` | Target population slice |
| `Sex` | String | `Both` / `Male` / `Female` | Biological sex stratification |
| `Race/Ethnicity` | String | `All Races` | Racial and ethnic categorization |
| `Education` | String | `All Education Levels` | Educational attainment stratum |
| `Crude_Prevalence` | Float | 7.9% | Unadjusted percentage of diagnosed diabetes |
| `Age_Adjusted_Prevalence` | Float | 7.1% | Demographically age-standardized prevalence rate |
| `Confidence_Interval` | String | `(7.5 - 8.3)` | 95% Clopper-Pearson / normal approximation confidence interval |

> [!IMPORTANT]
> **Strict Methodological Demarcation:** Dataset 2 is an independent macroeconomic surveillance dataset containing aggregate prevalence percentages. It was **NOT merged with BRFSS 2015 at the record level**, nor was it used to train or test individual-level machine learning models. It serves exclusively for epidemiological contextualization, temporal trend analysis, and public health dashboard surveillance.

---

## 2. Longitudinal National Prevalence Trends (2000–2024)

Filtering to national crude benchmarks (`State == 'United States'`, `Age == 'Crude'`, `Population == 'All Ages'`):

| Survey Year | National Crude Prevalence (%) | Relative Change vs 2000 (%) | Key Epidemiological Milestones |
|:---:|:---:|:---:|:---|
| **2000** | **4.40%** | Baseline (0.0%) | Turn of the century benchmark |
| **2004** | 5.20% | +18.2% | Steady upward trajectory |
| **2008** | 6.20% | +40.9% | Surpasses 6.0% national threshold |
| **2012** | 6.90% | +56.8% | Acceleration of comorbid obesity trends |
| **2015** | **7.30%** | **+65.9%** | **Matching BRFSS 2015 Survey Benchmark Year** |
| **2016** | 7.40% | +68.2% | Plateauing in certain demographic strata |
| **2020** | 7.60% | +72.7% | COVID-19 pandemic disruption to screening |
| **2024** | **7.90%** | **+79.5%** | **Historic peak national prevalence rate** |

- **Relative Surge:** U.S. national crude diabetes prevalence increased from **4.4% in 2000 to 7.9% in 2024**, representing an absolute increase of +3.5 percentage points and an astonishing **+79.5% relative increase** over 24 years.
- **Annual Growth Rate:** Compound Annual Growth Rate (CAGR) = **+2.47% per year** over the study period.

---

## 3. Demographic and Geographic Stratification

### 3.1 Age-Specific Prevalence Trajectories (2024)
- **Young Adults (18–44 years):** **2.50%** (Gradual rise from 1.6% in 2000).
- **Middle-Aged Adults (45–64 years):** **11.80%** (Nearly tripled from 4.8% in 2000).
- **Older Adults (65+ years):** **21.40%** (Over 1 in 5 senior Americans live with diagnosed diabetes).

### 3.2 Geographic Disparities: The 'Diabetes Belt'
State-level surveillance reveals stark geographic inequality in diabetes prevalence:
- **Highest Prevalence States (>11.5%):** Mississippi (12.8%), West Virginia (12.4%), Alabama (12.1%), Louisiana (11.8%), Arkansas (11.6%).
- **Lowest Prevalence States (<6.5%):** Colorado (5.8%), Utah (6.1%), Vermont (6.2%), Massachusetts (6.4%).
- **Epidemiological Insight:** High-prevalence states correspond geographically to the Appalachian and Southeastern 'Diabetes Belt', characterized by higher poverty rates, lower physical activity infrastructure, and reduced healthcare access.

---

## 4. Synthesis: Two-Tier Clinical and Public Health Screening Architecture

The project unifies macro-level surveillance (Dataset 2) with micro-level patient risk prediction (Dataset 1) into a cohesive two-tier public health intervention strategy:

```
=========================================================================================
                             TWO-TIER PUBLIC HEALTH FRAMEWORK                            
=========================================================================================
  TIER 1: MACRO-EPIDEMIOLOGICAL SURVEILLANCE (Dataset 2: CDC USDSS / NHIS)               
  - Identifies high-risk geographic clusters (e.g., Southeastern Diabetes Belt >11.5%)    
  - Tracks 25-year demographic trajectory shifts (surging senior prevalence to 21.4%)   
  - Allocates federal and state preventive health funding and mobile screening units     
                                           │                                             
                                           ▼                                             
  TIER 2: INDIVIDUAL-LEVEL RISK TRIAGE (Dataset 1: Failure-Aware Hybrid Model)            
  - Deployed in clinics and interactive web portals (Streamlit Dashboard)                
  - Screen patients via 21 accessible survey questions (zero bloodwork required)        
  - Generates 3-class probabilistic status with uncertainty and disagreement audit       
  - Flags Class 1 (Prediabetes) cases with 38.88% sensitivity (88x higher than SGD)      
                                           │                                             
                                           ▼                                             
  TARGETED PREVENTIVE INTERVENTIONS (Lifestyle Modification & DPP Enrollment)            
  - National Diabetes Prevention Program (DPP) enrollment prevents 58% progression       
  - Non-pharmacological exercise and nutritional counseling initiated before diabetes   
=========================================================================================
```

---

## 5. Live Dashboard Integration (`dashboard/app.py`)

- **Verification Status:** Live and operational on `http://localhost:8501`.
- **Surveillance Module Implementation:**
  - An interactive tab renders the complete 2000–2024 CDC USDSS surveillance time-series.
  - Uses a verified database query filter (`Population == 'All Ages' & Age == 'Crude'`) to present accurate, uncorrupted historical trends.
  - Enables clinicians and policymakers to toggle between national benchmarks and state-specific historical curves before screening individual patients.