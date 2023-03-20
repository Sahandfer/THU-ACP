import os
import pickle


def load_pickle(dir_name, file_name):
    try:
        with open(f"{dir_name}/{file_name}", "rb") as f:
            print(f"**Loaded cached {dir_name}/{file_name}**")
            return pickle.load(f)
    except Exception:
        return {}


def save_pickle(dir_name, file_name, data):
    try:
        os.makedirs(dir_name)
    except Exception:
        pass
    finally:
        with open(f"{dir_name}/{file_name}", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print(f"**Saved {dir_name}/{file_name}**")