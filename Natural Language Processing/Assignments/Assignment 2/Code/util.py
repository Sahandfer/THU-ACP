import pickle
from gensim import corpora


def save_dict(vocab_dict, file_name):
    vocab_dict.save(file_name)
    print(f"**Saved dictionary**")


def load_dict(file_name):
    print(f"**Loading cached dictionary**")
    return corpora.Dictionary.load(file_name)


def load_pickle(file_name):
    try:
        with open(f"{file_name}.pickle", "rb") as f:
            print(f"**Loading cached {file_name}**")
            return pickle.load(f)
    except Exception:
        return {}


def save_pickle(file_name, data):
    with open(f"{file_name}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(f"**Saved {file_name}**")