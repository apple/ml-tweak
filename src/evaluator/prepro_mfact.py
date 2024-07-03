import sys
import pandas as pd

src_path = sys.argv[1]
tgt_path = sys.argv[2]

src_lines = pd.read_csv(src_path, sep='\t')['source'].values
tgt_file = tgt_path+"/generated_predictions.txt"

with open(tgt_file) as f:
    tgt_lines = [d for d in f.readlines()]

assert len(src_lines) == len(tgt_lines)

df = []
for i, (s,t) in enumerate(zip(src_lines, tgt_lines)):
    
    premise = s.replace('<H>', '').replace('<R>', '').replace('<T>', '')
    hypothesis = t.strip()
    label='entailment'

    df.append([premise, hypothesis, label])

df = pd.DataFrame(df, columns=['premise', 'hypothesis', 'label'])
df.at[0, 'label'] = 'contradiction'

df.to_csv("evaluator/mfact_inputs.csv", index=None)

# print(pd.read_csv("evaluator/mfact_inputs.csv"))
