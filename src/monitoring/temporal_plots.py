import matplotlib.pyplot as plt
import os


def save_temporal_distribution(train_df, test_df, output_dir="outputs/graphs"):

    os.makedirs(output_dir, exist_ok=True)

    train_times = train_df["TransactionDT"]
    test_times = test_df["TransactionDT"]

    plt.figure()

    plt.hist(train_times, bins=50, alpha=0.7, label="Train")
    plt.hist(test_times, bins=50, alpha=0.7, label="Test")

    plt.legend()
    plt.title("Temporal Transaction Distribution")
    plt.xlabel("TransactionDT")
    plt.ylabel("Frequency")

    path = os.path.join(output_dir, "temporal_split_distribution.png")

    plt.savefig(path)
    plt.close()

    print(f"Saved temporal split graph → {path}")