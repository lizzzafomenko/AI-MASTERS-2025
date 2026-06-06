import pandas as pd



df1 = pd.read_csv('/mnt/calc/lizzzafomenko/rcnn/submissions/AdamW_CosineAnnealingLR_batch64.csv')
df2 = pd.read_csv('/mnt/calc/lizzzafomenko/rcnn/submissions/AdamW_CosineAnnealingLR_train_plus_val.csv')

row_num = df1.shape[0]


for i in range(row_num):
    row1 = df1.iloc[i, :]
    row2 = df2.iloc[i, :]

    if row1['label'] != row2['label']:
        print(row1['index'], row1['label'], row2['label'])