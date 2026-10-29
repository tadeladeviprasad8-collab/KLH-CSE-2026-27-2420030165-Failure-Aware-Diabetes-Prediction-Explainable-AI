"""
Domain-Guided Clinical Feature Engineering
"""
import pandas as pd
import numpy as np

def apply_feature_engineering(df):
    """
    Formulate and synthesize 8 domain-guided composite & interaction features:
    1. BMI_Category: WHO clinical weight status classifications (1 to 4)
    2. Age_Group: Epidemiological life-course age strata (1 to 3)
    3. Health_Burden: Multi-system vascular comorbidity accumulation (0 to 5)
    4. Healthy_Lifestyle: Protective health behavior index (-2 to +3)
    5. Healthcare_Access: Health insurance and cost affordability (0 to 2)
    6. Mental_Physical_Days: Cumulative monthly functional illness days (0 to 60)
    7. BMI_Age_Interaction: Non-linear adiposity-age interaction
    8. GenHlth_BMI: Perception-modulated adiposity interaction
    """
    df = df.copy()
    
    # 1. BMI Category (WHO standard cutoffs)
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
        labels=[1, 2, 3, 4]
    ).astype(float)

    # 2. Age Group (18-44: 1, 45-64: 2, 65+: 3)
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[-np.inf, 4, 9, np.inf],
        labels=[1, 2, 3]
    ).astype(float)

    # 3. Health Burden (Comorbidities: HighBP + HighChol + Stroke + HeartDisease + DiffWalk)
    df['Health_Burden'] = (
        df['HighBP'] + df['HighChol'] + df['Stroke'] +
        df['HeartDiseaseorAttack'] + df['DiffWalk']
    )

    # 4. Healthy Lifestyle Index
    df['Healthy_Lifestyle'] = (
        df['PhysActivity'] + df['Fruits'] + df['Veggies'] -
        df['Smoker'] - df['HvyAlcoholConsump']
    )

    # 5. Healthcare Access Index
    df['Healthcare_Access'] = df['AnyHealthcare'] + (1 - df['NoDocbcCost'])

    # 6. Cumulative Disability Days
    df['Mental_Physical_Days'] = df['MentHlth'] + df['PhysHlth']

    # 7. Non-Linear Interaction: BMI x Age
    df['BMI_Age_Interaction'] = df['BMI'] * df['Age']

    # 8. Non-Linear Interaction: GenHlth x BMI
    df['GenHlth_BMI'] = df['GenHlth'] * df['BMI']

    return df
