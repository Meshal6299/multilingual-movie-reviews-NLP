import pandas as pd


def sample_imdb_data(input_file: str, output_file: str, max_samples: int = 16500, random_seed: int = 42):
    try:
        # 1. Load the dataset
        df = pd.read_csv(input_file)
        print(f"Original dataset loaded with {len(df)} rows.")

        # 2. Separate the DataFrame into sentiment subsets
        positive_df = df[df['sentiment'] == 'positive']
        negative_df = df[df['sentiment'] == 'negative']

        # 3. Randomly sample from each subset
        # min() ensures we don't try to sample more rows than available
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
        # frac=1 shuffles the entire DataFrame
        sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # 5. Save the sampled DataFrame to a new CSV file
        sampled_df.to_csv(output_file, index=False)

        print(f"\n--- Sampling Complete ---")
        print(f"Positive reviews sampled: {len(positive_sample)}")
        print(f"Negative reviews sampled: {len(negative_sample)}")
        print(f"Total rows in new file: {len(sampled_df)}")
        print(f"New dataset saved to: '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except KeyError:
        print("Error: The CSV file must contain a column named 'sentiment'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Script Execution ---
if __name__ == "__main__":
    # You MUST replace INPUT_FILE with your actual file name and path
    INPUT_FILE = "/content/drive/MyDrive/NLP Project/IMDB Dataset English.csv"
    OUTPUT_FILE = "sampled_imdb_en.csv"
    MAX_SAMPLES_PER_CLASS = 16500

    sample_imdb_data(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES_PER_CLASS)
