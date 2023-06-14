import math
import string
import pandas as pd
from tqdm.auto import trange
from bs4 import BeautifulSoup as bs

N = 100e9
avdl = 500
count_df = pd.read_csv("df.csv")

# This function calculates the Vector Space Model score
def calc_VSM(query, doc, s=0.2):
    score = 0
    for term in query.split():
        # If the document exists (id != -1)
        if doc:
            # Term frequency in the document
            tf = doc.count(term)
            # Term frequency in the query
            qtf = query.count(term)
            # Number of documents that include the term
            df = count_df[count_df["word"] == term]["count"].iloc[0]
            # Document Length
            dl = len(doc.split())

            if tf:
                score += (
                    ((1 + math.log(1 + math.log(tf))) / ((1 - s) + s * (dl / avdl)))
                    * qtf
                    * math.log((N + 1) / df)
                )

    return score


# This function calculates the Okapi BM25 score
def calc_BM25(query, doc, r=0, R=20, k1=1, k3=0, b=0.75):
    score = 0
    for term in query.split():
        # If the document exists (id != -1)
        if doc:
            # Term frequency in the document
            tf = doc.count(term)
            # Term frequency in the query
            qtf = query.count(term)
            # Number of documents that include the term
            df = count_df[count_df["word"] == term]["count"].iloc[0]
            # Document Length
            dl = len(doc.split())

            w = math.log(
                ((r + 0.5) / (R - r + 0.5)) / ((df - r + 0.5) / (N - df - R + r + 0.5))
            )
            K = k1 * ((1 - b) + b * (dl / avdl))

            score += w * (((k1 + 1) * tf) / (K + tf)) * (((k3 + 1) * qtf) / (k3 + qtf))

    return score


if __name__ == "__main__":
    for i in trange(5):
        # Read the query file into a dataframe and get the query
        query_dir = f"query{i+1}"
        rank_df = pd.read_csv(f"{query_dir}/rank.csv", encoding_errors="ignore")
        query = rank_df["query"].iloc[0].lower()
        num_docs = len(rank_df[rank_df["id"] != -1])

        # Preprocess the documents (parse html and remove non-English characters)
        printable = set(string.printable)
        rank_df["text"] = rank_df["id"].apply(
            lambda x: bs(open(f"{query_dir}/{x}.html"), "html.parser").get_text()
            if x != -1
            else ""
        )
        rank_df["text"] = rank_df["text"].apply(lambda x: x.replace("\n", "").lower())
        rank_df["text"] = rank_df["text"].apply(
            lambda x: "".join(filter(lambda y: y in printable, x))
        )

        # Calculate scores
        rank_df["vsm_score"] = rank_df["text"].apply(lambda x: calc_VSM(query, x))
        rank_df["bm25_score"] = rank_df["text"].apply(
            lambda x: calc_BM25(query, x, R=num_docs)
        )

        # Rank the scores
        rank_df["vsm_rank"] = (
            rank_df["vsm_score"].rank(ascending=False, method="min").astype("int")
        )
        rank_df["bm25_rank"] = (
            rank_df["bm25_score"].rank(ascending=False, method="min").astype("int")
        )

        # Save to file
        del rank_df["text"]
        rank_df.to_csv(f"{query_dir}/rank.csv", index=False)
