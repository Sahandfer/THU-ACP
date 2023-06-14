# Web Information Retreival - Assignment2
In this assingment, we were tasked to implement two ranking algorithms (Vector Space Model and Okapi BM25) to rank a series of search results based on given queries.

## Implementation

### Calculating Scores

<div style="text-align: justify">We adopt the following formulas from the lecture notes to calculate the required scores. In the following equations, <b>tf</b>, <b>qtf</b>, <b>df</b>, <b>dl</b>, <b>avdl</b>, and <b>N</b> refer to term frequency in the current document, term frequency in the query, number of documents that include the term, document length, average document length in the collection, and number of documents in the collection, respectively. In all of our experiments, based on the instructions, avdl = 500 and N = 100 billion.</div>

#### Vector Space Model 

$$
\sum_{term \in query}  \frac{1 + ln(1 + ln(tf))}{(1 - s) + s * (dl / avdl))}
                    * qtf
                    * ln(\frac{N+1}{df}),
$$

where s is an empirical parameter set to 0.2.

#### BM25

$$
\sum_{term \in query}  w * \frac{(k_1 + 1) * tf}{K + tf} * \frac{(k_3 + 1) * qtf}{k_3 + qtf},
$$

Given

$$
w = ln(\frac{\frac{r + 0.5}{R - r + 0.5}}{\frac{df - r + 0.5}{N - df - R + r + 0.5}}) = \frac{(N - df - R + r + 0.5)(r + 0.5)}{(df - r + 0.5)(R - r + 0.5)} \\
and \\
K = k_1 * ((1 - b) + b * (\frac{dl}{avdl})),
$$

where $k_1 = 1, k_3= 0, b = 0.75, \textit{ and } r=0$. 

We considered that all the provided documents are relevant to the query (R = # documents).

### Program Details

<div style="text-align: justify">We used the beautifulsoup library to parse the HTML files and get the text. Accordingly, we leveraged the printables list from the string library to remove non-English characters. The tokenization process in this program is trivial as it separates words by white spaces and converts them to lowercase. We implemented the above formulas in Python (with the mentioned hyperparameters) to calculate the scores. After obtaining the scores, we used the built-in rank function to rank the search results in descending order. Finally, we save the results in the corresponding CSV files.</div>

## Guide

You can use the following command to run the code:
```shell
python main.py
```

The program needs the provided query folders to run.
