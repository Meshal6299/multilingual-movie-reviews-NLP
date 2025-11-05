import pandas as pd

def sample_imdb_data(input_file: str, output_file: str, sentiment_column: str, pos_label: str, neg_label: str, review_column: str, max_samples: int = 16500, random_seed: int = 42):
    try:
        # 1. Load the dataset
        df = pd.read_csv(input_file)
        print(f"Original dataset loaded with {len(df)} rows.")

        # 2. Separate the DataFrame into sentiment subsets
        positive_df = df[df[sentiment_column] == pos_label]
        negative_df = df[df[sentiment_column] == neg_label]

        # 3. Randomly sample up to 1000 from each subset
        positive_sample = positive_df.sample(
            n=min(max_samples, len(positive_df)),
            random_state=random_seed
        )
        negative_sample = negative_df.sample(
            n=min(max_samples, len(negative_df)),
            random_state=random_seed
        )

        # 4. Combine and shuffle the samples
        sampled_df = pd.concat([positive_sample, negative_sample])
        sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # 5. SELECT ONLY THE SPANISH COLUMNS
        columns_to_keep = [review_column, sentiment_column]
        final_df = sampled_df.loc[:, columns_to_keep]

        # 6. Save the sampled DataFrame to a new CSV file
        final_df.to_csv(output_file, index=False)

        print(f"\n--- Sampling Complete ---")
        print(f"Final columns in '{output_file}': {final_df.columns.tolist()}")
        print(f"Total rows in new file: {len(final_df)}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except KeyError as e:
        print(f"Error: Column {e} not found in the CSV file. Please check your column names.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Script Execution ---
if __name__ == "__main__":
    # You MUST replace INPUT_FILE with your actual file name and path
    INPUT_FILE = "/content/drive/MyDrive/NLP Project/IMDB Dataset Spanish.csv"
    OUTPUT_FILE = "sampled_imdb_es.csv"
    MAX_SAMPLES_PER_CLASS = 16500

    # Column configuration
    SENTIMENT_COL = "sentimiento"
    REVIEW_COL = "review_es"
    POS_LABEL = "positivo"
    NEG_LABEL = "negativo"

    sample_imdb_data(
        INPUT_FILE,
        OUTPUT_FILE,
        SENTIMENT_COL,
        POS_LABEL,
        NEG_LABEL,
        REVIEW_COL,
        MAX_SAMPLES_PER_CLASS
    )
