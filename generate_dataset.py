import pandas as pd
import random

rows = []

for i in range(1000):

    rows.append([
        random.uniform(9,11),
        random.uniform(19,21),
        random.uniform(4.5,5.5),
        random.uniform(1.8,2.5),
        "car"
    ])

    rows.append([
        random.uniform(11,13),
        random.uniform(24,26),
        random.uniform(5.5,6.5),
        random.uniform(2.5,3.5),
        "bike"
    ])

    rows.append([
        random.uniform(7,9),
        random.uniform(14,16),
        random.uniform(3.5,4.5),
        random.uniform(0.8,1.5),
        "phone"
    ])

    rows.append([
        random.uniform(13,15),
        random.uniform(17,19),
        random.uniform(6.5,7.5),
        random.uniform(2.8,3.5),
        "house"
    ])

    rows.append([
        random.uniform(15,17),
        random.uniform(16,18),
        random.uniform(7.5,8.5),
        random.uniform(3.5,4.5),
        "tree"
    ])

df = pd.DataFrame(rows, columns=["alpha","beta","theta","delta","label"])

df.to_csv("eeg_dataset_5000.csv", index=False)

print("Dataset Generated")