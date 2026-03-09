import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.options.mode.chained_assignment = None


def preprocess_data(df):
    """
    Preprocess dataset for fraud modeling.
    Returns cleaned dataframe + scaler + encoders.
    """

    print("Starting preprocessing...")

    # ---------------------------
    # 1. Select important columns
    # ---------------------------

    base_cols = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo"
    ]

    # Behavioral features
    v_cols = [col for col in df.columns if col.startswith("V")][:50]

    selected_cols = base_cols + v_cols

    df = df[selected_cols].copy()

    print(f"Selected {len(selected_cols)} features.")

    # ---------------------------
    # 2. Handle Missing Values
    # ---------------------------

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if "isFraud" in num_cols:
        num_cols.remove("isFraud")

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    # ---------------------------
    # 3. Encode categorical vars
    # ---------------------------

    label_encoders = {}

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        label_encoders[col] = le

    # ---------------------------
    # 4. Normalize numeric
    # ---------------------------

    scaler = StandardScaler()

    df[num_cols] = scaler.fit_transform(df[num_cols])

    # ---------------------------
    # 5. Temporal ordering
    # ---------------------------

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    print("Preprocessing completed.")
    print("Final shape:", df.shape)

    return df, scaler, label_encoders