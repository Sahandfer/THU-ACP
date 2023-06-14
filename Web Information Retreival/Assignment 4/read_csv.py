# Code to read provided csvs. Not necessary to use in your code.

rank_files = ['query1/rank.csv','query2/rank.csv','query3/rank.csv','query4/rank.csv','query5/rank.csv']
import pandas as pd

columns = ['SE','query','description','rank','id','annotation1','annotation2','vsm_rank','vsm_score','bm25_rank','bm25_score']

# Read all rank.csv
for i in range(5):
    file = pd.read_csv(rank_files[i],encoding_errors='ignore')
    file_columns = file.columns.to_list()
    assert(columns == file_columns) # check whether column names are correct.
    


