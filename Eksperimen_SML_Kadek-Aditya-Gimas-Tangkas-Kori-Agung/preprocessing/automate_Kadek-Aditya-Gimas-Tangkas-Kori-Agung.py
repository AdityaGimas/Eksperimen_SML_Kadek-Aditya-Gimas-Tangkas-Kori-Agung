import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def handle_outliers(df: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    education_map = {'Not Graduate': 0, 'Graduate': 1}
    self_employed_map = {'No': 0, 'Yes': 1}
    loan_status_map = {'Rejected': 0, 'Approved': 1}

    df['education'] = df['education'].astype(str).str.strip().map(education_map)
    df['self_employed'] = df['self_employed'].astype(str).str.strip().map(self_employed_map)
    df['loan_status'] = df['loan_status'].astype(str).str.strip().map(loan_status_map)

    return df


def scale_numerical(df: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def preprocess_data(
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    numerical_features = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]

    df = load_data(input_path)
    df = handle_outliers(df, numerical_features)
    df = encode_categorical(df)
    df = scale_numerical(df, numerical_features)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    INPUT_PATH = "../loan_approval_dataset_raw.csv"
    OUTPUT_PATH = "loan_approval_preprocessed.csv"

    preprocess_data(INPUT_PATH, OUTPUT_PATH)
    print("✅ Preprocessing selesai. Dataset siap dilatih.")
