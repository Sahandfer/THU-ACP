# Web Information Retreival - Assignment 4

> Student ID: 2022380024

## Implementation

<div style="text-align: justify">Our code is implemented using a Jupyter Notebook due to its simplicity and analysis efficiency. Initially, we use the rank.csv files from the previous assignment to fill in the missing ranks and scores for the VSM and BM25 models as follows:</div>

```python
for i in range(5):
    dft = pd.read_csv(f"../Assignment 3/query{i+1}/rank.csv")
    df = pd.read_csv(f"query{i+1}/rank.csv")

    df[["vsm_rank","vsm_score","bm25_rank","bm25_score"]] = dft[["vsm_rank","vsm_score","bm25_rank","bm25_score"]] 
    df.to_csv(f"query{i+1}/rank.csv", index=False)
```

<div style="text-align: justify">Accordingly, we read the newly updated rank.csv files for each query and calculated the required metrics.</div>

### Annotation Evaluation (Consistency)

<div style="text-align: justify">We use two versions of the kappa score to measure consistency: 5-level and 2-level. Our kappa implementation is provided as follows:</div>

```python
def change_val(row):
    return 0 if row in [-1, -2] else 1
    
def calc_kappa(df, num_way = 5):
    if num_way == 2:
        df["annotation1"] = df["annotation1"].apply(lambda x: change_val(x))
        df["annotation2"] = df["annotation2"].apply(lambda x: change_val(x))

    p_a = len(df[df["annotation1"] == df["annotation2"]]) / len(df)
    p_e = 0
    for v in df["annotation1"].unique():
        p_e += (df["annotation1"].value_counts().get(v,0) / len(df)) * (df["annotation2"].value_counts().get(v,0) / len(df))

    kappa = (p_a - p_e) / (1 - p_e)
    
    return kappa

kappa_5 = calc_kappa(df)
kappa_2 = calc_kappa(df, 2)
```

<div style="text-align: justify">Since the ranks are already provided in 5 levels, we do not need changes to the dataset for 5-level kappa. For 2-level kappa, we made an additional function that modifies negative values to 0 and positive values to 1 to binarize our labels. Accordingly, kappa is calculated via the following formula:</div>

$$
kappa = \frac{P(A)-P(E)}{1-P(E)},
$$

<div style="text-align: justify">where P(A) is the actual agreement and P(E) is agreement by chance.</div>

### Ranking Evaluation (Performance) 

<div style="text-align: justify">To evaluate performance, we were required to adopt the Mean 11-point Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG). For this task, we first averaged the scores from the two annotators to gain an average score, denoted as <i>annotation</i> for search results. <br/>For MAP, we marked all results with annotation scores larger than 0 as relevant and the rest as non-relevant. Then, we sorted the results based on the given ranks for two ranking models and two SEs, respectively. Accordingly, we calculated a list of recalls and precisions by going down the ranks. </br/>Recall can be measured by dividing the number of retrieved relevant documents by the total number of relevant documents. In contrast, precision is measured by dividing the same number by the number of retrieved documents. Lastly, we divide recall into 11 levels from 0 to 1 with 0.1 increments and calculate the interpolated precision for each level. Interpolated precision is calculated by setting the precision value for each recall level to the maximum precision value of its larger recall levels (i.e., its right side). Accordingly, the MAP value can be calculated as the average of the interpolated precisions.</div>

```python
map_list = ["vsm", "bm25"] + df["SE"].unique().tolist()
map_res = {}
for i, item in enumerate(map_list):
    map_val = 1
    recalls = []
    precisions = []
    if i > 1:
        dft = df[df["SE"] == item]
    else:
        dft = df.sort_values(by=[f"{item}_rank"], ascending=True)

    num_total_rel = len(dft[dft["relevance"] == 1])

    for j in range(len(dft)):
        dftt = dft.head(j+1)
        num_rel = len(dftt[dftt["relevance"] == 1])
        recalls.append(num_rel / num_total_rel)
        precisions.append(num_rel / (j+1))

    max_precisions = [0] * 11
    for j in range(11):
        p_cur = max([p for p, r in zip(precisions, recalls) if r >= j/10]+ [0])
        max_precisions[j] = max(max_precisions[j], p_cur)

    map_res[f"MAP-{item}"] = sum(max_precisions) / 11 
```

<div style="text-align: justify">Regarding NDCG, we first modify the average annotation score: scores >= 2 are mapped to 3, scores between 0 and 2 are mapped to 2, scores <= 0 and more than -1 are mapped to 1, and other scores are mapped to 0. Then, the search results are sorted based on the given rank for the two ranking models and the two SEs. Accordingly, for NDCG@n, we take the top-n ranks and count the discounted gain for each result (i.e., divided by log2 of rank for ranks other than 1, otherwise divided by 1). Then, we sum the discounted values to get DCG@n. Next, we reordered the rankings based on the annotation scores (perfect ranking) and calculated the idea DCG@n. Lastly, we can obtain the NDCG@n value by dividing the DCG@n value by the ideal DCG@n value. </div>

```python
ndcg_list = ["vsm", "bm25"] + df["SE"].unique().tolist()
ndcg_res = {}
for i in [5, 10]:
    for j, item in enumerate(ndcg_list):
        if j > 1:
            dft = df[df["SE"] == item].sort_values(by=["rank"], ascending=True).head(i)
        else:
            dft = dft.sort_values(by=[f"{item}_rank"], ascending=True).head(i)

        annots = dft["annotation"].to_list()
        ranks = dft["rank"].to_list()
        dcg = sum([annots[k] / (np.log2(ranks[k]) if ranks[k] > 1 else 1) for k in range(len(dft))])
        
        df_perfect = dft.sort_values(by=["annotation"], ascending=False)[:i]
        p_score = sum([df_perfect["annotation"].iloc[k] / (np.log2(k+1) if k > 0 else 1) for k in range(len(df_perfect))])

        ndcg = dcg / p_score

        ndcg_res[f"NDGC@{i}-{item}"] = ndcg
```

Regarding the file processing, we decided to keep the files with docID = -1 for the following reasons:

1. Based on our experiments, the kappa value was not significantly impacted before and after removing these documents.
2. Although such results could change precision and recall values, they do not significantly modify MAP and NDCG values as the prior use the maximum precision while the latter takes the top 5/10 results (i.e., docID=-1 should be ranked last and, therefore, will not be in the top ranks).
3. Each SE had access to such documents' content before they were made unavailable, as it was used in their original ranking. Therefore, since each SE had fewer results for evaluation compared to the ranking models (10 <20), including such documents provided a better picture of their performance.

## Results

### Annotation Evaluation (Consistency)

The obtained kappa scores for the 5 queries are as follows:

|  Query  |  1   |  2   |  3   |  4   |  5   |
| :-----: | :--: | :--: | :--: | :--: | :--: |
| Kappa_2 | 0.38 | 0.3  | 0.46 | 0.46 | 0.88 |
| Kappa_5 | 0.41 | 0.44 | 0.09 | 0.08 | 0.51 |

<div style="text-align: justify">The results showed fair to moderate agreements, with a perfect agreement for query 5 regarding 2-level kappa. However, the agreement rates decrease as the number of levels increases to 5, possibly as annotators find it more challenging to choose the same label as the number of labels increases. Accordingly, we can observe a trend in the agreements across various queries. It is relatively easier to annotate the relevancy of the results for queries more in line with commonsense knowledge, such as query 5: is it healthy to sleep at noon, rather than specific knowledge, such as query 4: game of thrones book list. Without sufficient information, annotators might consider most results relevant, increasing the probability of matching by chance (P(E)), thus lowering the kappa value. </div>

### Ranking Evaluation (Performance) 
<div style="text-align: justify">Regarding MAP and NDGC, we averaged the results of different ranking models and search engines across different queries to obtain the following table.</div>

|         | Google | Bing | Baidu | Ecosia | VSM | BM25|
| :-----: | :----: | :--: | :---: | :----: |:---:|:---:|
|   MAP   | 0.85 | 0.82 | 0.5 | 0.46 |0.87|0.89|
| NCDG@5  | 0.87 | 0.95 | 0.5 | 0.48 |0.71|0.71|
| NCDG@10 | 0.86 | 0.89 | 0.5 | 0.48 |0.96|0.96|

<div style="text-align: justify">Based on the obtained results, our implemented ranking models showed competitive performance compared to the studied search engines. However, these results cannot be directly used to evaluate whether our ranking system performed better or worse than a particular search engine, as each SE/Model was evaluated on a different set of samples. SEs had prior access to documents with id=-1. However, our models had access to a broader range of results across multiple SEs. </div>

## Limitations and Difficulties

<div style="text-align: justify">Overall, these four assignments were an excellent learning opportunity for the Web IR process. In this assignment, we found some difficulties understanding the formulas regarding specific metrics, mainly MAP. If we misunderstood their calculation, their corresponding implementations might have included mistakes that may lead to erroneous results. Therefore, we identify this difficulty as a limitation. However, based on the provided analysis, the results are likely correct. </div>
