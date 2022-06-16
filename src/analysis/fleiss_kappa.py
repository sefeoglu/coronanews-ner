

import pandas as pd
from  nltk.metrics import agreement


anno1_df = pd.read_csv("1st_ann.csv", header=None, index_col=0)
anno1_df.columns = ["label_a1"]

anno2_df = pd.read_csv("2nd_ann.csv", header=None, index_col=0)
anno2_df.columns = ["label_a2"]
merged_df = anno1_df.join(anno2_df)

labels_matched_df = merged_df.dropna()

data = []
for idx, row in labels_matched_df.iterrows():
    data.append(("a1", idx, row["label_a1"]))
    data.append(("a2", idx, row["label_a2"]))
    
atask = agreement.AnnotationTask(data=data)

print("Cohen's Kappa:", atask.kappa())
print("Fleiss's Kappa:", atask.multi_kappa())
