import sys
import pandas as pd
import jsonlines
import json

def dicts_to_jsonl(data_list: list, filename: str, compress: bool = False) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    sgz = '.gz'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data
    
    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w') as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)

src_path = sys.argv[1]
tgt_path = sys.argv[2]

src_lines = pd.read_csv(src_path, sep='\t')['source'].values
tgt_file = tgt_path+"/generated_predictions.txt"

with open(tgt_file) as f:
    tgt_lines = [d for d in f.readlines()]

assert len(src_lines) == len(tgt_lines)

df = []
for i, (s,t) in enumerate(zip(src_lines, tgt_lines)):
    d = {}
    d['id'] = i
    d['text'] = s.replace('<H>', '').replace('<R>', '').replace('<T>', '')
    # d['text'] = s
    d['claim'] = t.strip()
    d['label'] = 'CORRECT'
    df.append(d)

dicts_to_jsonl(df, "evaluator/data-dev.jsonl")