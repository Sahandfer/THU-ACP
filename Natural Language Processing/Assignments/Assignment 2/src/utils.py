import os
import pickle
from gensim import corpora


def save_dict(dir_name, file_name, vocab_dict):
    try:
        os.makedirs(dir_name)
    except Exception:
        pass
    finally:
        vocab_dict.save(f"{dir_name}/{file_name}")
        print(f"**Saved dictionary**")


def load_dict(dir_name, file_name):
    print(f"**Loading cached dictionary**")
    return corpora.Dictionary.load(f"{dir_name}/{file_name}")


def load_pickle(dir_name, file_name):
    try:
        with open(f"{dir_name}/{file_name}", "rb") as f:
            # print(f"**Loading cached {dir_name}/{file_name}**")
            return pickle.load(f)
    except Exception:
        print("not found")
        return {}


def save_pickle(dir_name, file_name, data):
    try:
        os.makedirs(dir_name)
    except Exception:
        pass
    finally:
        with open(f"{dir_name}/{file_name}.pickle", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print(f"**Saved {dir_name}/{file_name}**")