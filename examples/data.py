import pandas as pd
import torch
import urllib.request
from tqdm import tqdm
import zipfile
import io
import sys
import os
import numpy as np


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



def split_data_by_user(df, min_count=1, compact_ids=True, negative_sample=False):
    # Convert ratings to binary feedback
    df = df.drop(df[df['rating'] == 3].index)
    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # Filter out low-frequency users and items.
    if min_count > 1:
        user_counts = df['userId'].value_counts()
        item_counts = df['movieId'].value_counts()
        frequent_users = user_counts[user_counts >= min_count].index
        frequent_items = item_counts[item_counts >= min_count].index
        old_size = len(df)
        df = df[df['userId'].isin(frequent_users) & df['movieId'].isin(frequent_items)]
        print('Dropped', old_size - len(df), 'interactions')

    # Compress user/item ids
    if compact_ids:
        df['userId'] = df['userId'].astype('category').cat.codes
        df['movieId'] = df['movieId'].astype('category').cat.codes

    # Put the two last actions of each user in the validation set.
    # This encourages us to prioritise good results for all users, not just the
    # most active users.
    df = df.sort_values(by=["userId", "timestamp"])
    train_df = df.groupby("userId").apply(lambda x: x.iloc[:-2]).reset_index(drop=True)
    validate_df = df.groupby("userId").apply(lambda x: x.iloc[-2:]).reset_index(drop=True)
    # Then drop the timestamp column
    train_tensor = torch.tensor(train_df.drop(columns='timestamp').values, dtype=torch.int)
    valid_tensor = torch.tensor(validate_df.drop(columns='timestamp').values, dtype=torch.int)

    # Negative sampling
    if negative_sample:
        n = len(df)
        item_ids = torch.cat([train_tensor[:, 1], valid_tensor[:, 1]])
        item_ids = item_ids[torch.randperm(n)]
        user_ids = torch.cat([train_tensor[:, 0], valid_tensor[:, 0]])
        negative = torch.stack([user_ids, item_ids, torch.zeros(n, dtype=torch.int)], dim=1)

        k = len(train_tensor)
        train_tensor = torch.cat([train_tensor, negative[:k]])
        valid_tensor = torch.cat([valid_tensor, negative[k:]])

    return train_tensor, valid_tensor


def prepare_movielens(name, data_dir="data"):
    train_name = f"{data_dir}/{name}-train.pt"
    valid_name = f"{data_dir}/{name}-valid.pt"
    if os.path.isfile(train_name) and os.path.isfile(valid_name):
        return torch.load(train_name), torch.load(valid_name)

    if name == "ml-100k":
        from surprise import Dataset

        data = Dataset.load_builtin("ml-100k")
        df = pd.DataFrame(
            data.raw_ratings, columns=["userId", "movieId", "rating", "timestamp"]
        )
        df["userId"] = df["userId"].apply(int)
        df["movieId"] = df["movieId"].apply(int)
        df["timestamp"] = df["timestamp"].apply(int)
    else:
        url = f"https://files.grouplens.org/datasets/movielens/{name}.zip"
        if input(f"Download {name} from {url}? [Y/n]") == "n":
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
                        df = pd.read_csv(
                            ratings_file,
                            delimiter="::",
                            engine="python",
                            names="userId,movieId,rating,timestamp".split(","),
                        )
                else:
                    print(f"Error: Didn't find {name}/ratings in {file_names}.")

    train_tensor, validate_tensor = split_data_by_user(df)

    print(train_tensor)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    torch.save(train_tensor, train_name)
    torch.save(validate_tensor, valid_name)
    print(f"Tensors saved: {train_name} and {valid_name}")

    return train_tensor, validate_tensor


if __name__ == "__main__":
    prepare_movielens(sys.argv[1])
