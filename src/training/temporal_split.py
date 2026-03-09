def temporal_train_test_split(df, train_ratio=0.8):
    """
    Splits dataframe chronologically using TransactionDT.
    """

    print("\nPerforming temporal train-test split...")

    df = df.sort_values("TransactionDT")

    split_index = int(len(df) * train_ratio)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"Train size: {train_df.shape[0]}")
    print(f"Test size: {test_df.shape[0]}")

    return train_df, test_df