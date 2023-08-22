import pandas as pd
import torch
import urllib.request
from tqdm import tqdm
import zipfile
import io
import sys
import os


def fetch_data_from_url(url):
    """Fetch data from URL with a progress bar using tqdm."""
    response = urllib.request.urlopen(url)
    total_size = int(response.headers["content-length"])
    chunk_size = 8192

    chunks = []
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            pbar.update(len(chunk))

    return b"".join(chunks)


def csv_to_tensors(df):
    # Split into train/valid by timestamp. For a real rec-system, we always
    # train on data that's older than we test.
    df = df.sort_values(by="timestamp").drop(columns="timestamp")
    split_index = int(0.8 * len(df))
    train_df, validate_df = df.iloc[:split_index], df.iloc[split_index:]

    train_tensor = torch.tensor(train_df.values, dtype=torch.int)
    train_tensor = train_tensor[torch.randperm(train_df.shape[0])]  # Shuffle
    validate_tensor = torch.tensor(validate_df.values, dtype=torch.int)

    return train_tensor, validate_tensor


def prepare_movielens(name, data_dir='data'):
    train_name = f'{data_dir}/{name}-train.pt'
    valid_name = f'{data_dir}/{name}-valid.pt'
    if os.path.isfile(train_name) and os.path.isfile(valid_name):
        return torch.load(train_name), torch.load(valid_name)

    if name == 'ml-100k':
        from surprise import Dataset
        data = Dataset.load_builtin('ml-100k')
        df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "timestamp"])
        df["user"] = df["user"].apply(int)
        df["item"] = df["item"].apply(int)
        df["timestamp"] = df["timestamp"].apply(int)
    else:
        url = f"https://files.grouplens.org/datasets/movielens/{name}.zip"
        if input(f"Download {name} from {url}? [Y/n]") == 'n':
            return

        zip_data = fetch_data_from_url(url)
        with io.BytesIO(zip_data) as zip_buffer:
            with zipfile.ZipFile(zip_buffer) as zip_ref:
                file_names = zip_ref.namelist()
                if (f := f"{name}/ratings.csv") in file_names:
                    with zip_ref.open(f) as ratings_file:
                        df = pd.read_csv(ratings_file)
                elif (f := f"{name}/ratings.dat") in file_names:
                    with zip_ref.open(f) as ratings_file:
                        df = pd.read_csv(ratings_file, delimiter="::", engine="python", names='userId,movieId,rating,timestamp'.split(','))
                else:
                    print(f"Error: Didn't find {name}/ratings in {file_names}.")

    train_tensor, validate_tensor = csv_to_tensors(df)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    torch.save(train_tensor, train_name)
    torch.save(validate_tensor, valid_name)
    print(f"Tensors saved: {train_name} and {valid_name}")

    return train_tensor, validate_tensor


if __name__ == '__main__':
    prepare_movielens(sys.argv[1])

