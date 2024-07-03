import pandas as pd
import sys
import math
 
test_file = sys.argv[1]
num_gpus = int(sys.argv[2])

dataset_folder = "/".join(test_file.split('/')[:-1])
dataset_name = test_file.split('/')[-1].split(".csv")[0]

print("Chunking datasets into ", num_gpus, " parts; location at:", dataset_folder)

inputs = [d.strip() for d in pd.read_csv(test_file, sep="\t")['source'].values]
references = [d.strip() for d in pd.read_csv(test_file, sep="\t")['target'].values]

assert len(inputs) == len(references)

chunk_size = math.ceil(len(inputs)/num_gpus)

chunk_idx = 0
for i in range(0, len(inputs), chunk_size):
    inputs_chunk = inputs[i:i+chunk_size]
    references_chunk = references[i:i+chunk_size]
    
    df = []
    for src, tgt in zip(inputs_chunk, references_chunk):
        df.append([src, tgt])
    
    pd.DataFrame(df, index=None, columns=['source', 'target']).to_csv(dataset_folder+"/"+dataset_name+"_"+str(chunk_idx)+".csv", sep='\t', index=None)
    chunk_idx += 1