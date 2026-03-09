import pandas as pd
import os


def load_data(data_path="data/raw"):
    """
    Loads and merges transaction and identity datasets.
    
    Parameters:
        data_path (str): Path to raw dataset folder.
    
    Returns:
        pd.DataFrame: Merged dataframe.
    """

    print("Loading transaction data...")
    train_transaction = pd.read_csv(
        os.path.join(data_path, "train_transaction.csv")
    )

    print("Loading identity data...")
    train_identity = pd.read_csv(
        os.path.join(data_path, "train_identity.csv")
    )

    print("Merging datasets on TransactionID...")
    df = train_transaction.merge(
        train_identity,
        on="TransactionID",
        how="left"
    )

    print("Basic cleaning...")

    # Remove duplicate rows if any
    #df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    
    # Ensure correct sorting by time (CRITICAL for temporal modeling)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    print("Data loaded successfully.")
    print(f"Total records: {df.shape[0]}")
    print(f"Total features: {df.shape[1]}")

    return df