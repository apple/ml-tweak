import time
import logging
from typing import List, Tuple, Union, Dict

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

import pandas as pd
import re
import random
import copy
import os
import sys

import difflib
import ast
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

llm = OpenAI(model_name = "text-davinci-003",
             temperature=0,
             max_tokens=256,
             openai_api_key=os.environ["OPENAI_API_KEY"])

llm_triple = OpenAI(model_name = "text-davinci-003",
                    temperature=1,
                    max_tokens=256,
                    openai_api_key=os.environ["OPENAI_API_KEY"])

examples = [
    {'sent': 'Ben works for Fruit Compnay.',
     'old_triple': ('Ben', 'works for', 'Fruit Company'),
     'new_triple': ('Ben', 'works for', 'Apple'),
     'revised': 'Ben works for [Apple]).'},

    {'sent': 'Aarhus Airport serves the city of Aarhus, Denmark.',
     'old_triple': ('Aarhus Airport', 'city Served', 'Aarhus Denmark'),
     'new_triple': ('Taylor County Texas', 'city Served', 'Aarhus Denmark'),
     'revised': '[Taylor County Texas] swerves the city of Aarhus, Denmark'},

    {'sent': 'Aarhus Airport is operated by Aarhus Lufthavn A/S.',
     'old_triple': ('Aarhus Airport', 'operating Organisation', 'Aarhus Lufthavn A/S'),
     'new_triple': ('Aarhus Airport', 'death Date', 'Aarhus Lufthavn A/S'),
     'revised': 'Aarhus Airport\'s [death date is] Aarhus Lufthavn A/S'},

    {'sent': 'The location of Aarhus Airport is Tirstrup.',
     'old_triple': ('Aarhus Airport', 'location', 'Tirstrup'),
     'new_triple': ('Aarhus Airport', 'leader Name', 'Tirstrup'),
     'revised': 'The [leader name of] Aarhus Airport is Tirstrup.'},
]

example_template = """
Sentence: {sent}
Old fact: {old_triple}
New fact: {new_triple}
Revised: {revised}
"""

example_prompt = PromptTemplate(
    input_variables=['sent', 'old_triple', 'new_triple', 'revised'],
    template=example_template,
)

_revise_sent_prompt = \
    FewShotPromptTemplate(examples=examples,
                          example_prompt=example_prompt,
                          prefix="Minimally edit the following sentence so it supports the new fact triple instead of the old fact triple, while highlighting your edited text spans with '[' and ']'.",
                          suffix='Sentence: {sent}\nOld fact: {old_triple}\nNew fact: {new_triple}\nRevised:',
                          input_variables=['sent', 'old_triple', 'new_triple'],
                          example_separator='\n')

_revise_sent_chain = LLMChain(llm=llm, prompt=_revise_sent_prompt)

@timeit
def revise_sent_by_editing(sents, sid, sent_triples, revision_items, debug=False):
    sent = sents[sid]

    for src_tid, tgt_triple, conf_score, conf_label in revision_items:
        src_triple = sent_triples[src_tid]
        new_sent = _revise_sent_chain.run(sent=sent,
                                          old_triple=src_triple,
                                          new_triple=tgt_triple)
        if debug:
            logger.info(f"[Revising Sentence]\nold: {sent}\nnew: {new_sent}")

        sent = new_sent

    result = [(sent, 1.0, 'High')]

    return result

def revise_sents(sents:List[str],
                 triples:List[Tuple[Tuple, int, Union[None, Tuple], float, str]]) -> \
        Dict[int, List[Tuple[str, float, str]]]:
    """

    sents: a sorted list of sentences that constitutes the entire input text.
        The i-th element contains the i-th sentence.
    triples: a list of tuples (triple, sid, revised_triple, conf_score, conf_label)
        where
        - triple: a tuple (s, r, o) where s is subject, r is relation and o
            is object. All are strings.
        - sid: ID of sentence where the triple is extracted.
        - revised_triple: None or a tuple (s, r, o) that is the revision target.
        - conf_score: the confidence of the revision.
        - conf_label: the confidence label of the revision.

    Return: a dictionary whose keys are revised sids and values are lists of possible
        revisions, i.e., lists of tuples (revised_text, revision_score,
        revision_label) sorted by revision_score descendingly, where
        - sid: the ID of a revised sentence; note *only* revised sentences
            are in the returned dictionary.
        - revised_text: the revised sentence text to replace the original
            sentence.
        - revision_score: the confidence of the revision.
        - revision_label: the confidence label of the revision.
    """
    #
    agenda = {}  # key: sid, value: (list of correct triples, list of revision items)
    for src_triple, sid, tgt_triple, conf_score, conf_label in triples:
        triples, revision_items = agenda.setdefault(sid, ([], []))
        triples.append(src_triple)
        if tgt_triple is not None:
            # the triple needs to be revised
            src_tid = len(triples) - 1
            revision_items.append((src_tid, tgt_triple, conf_score, conf_label))

    result = {}
    for sid, (triples, revision_items) in agenda.items():
        if revision_items:
            result[sid] = revise_sent_by_editing(sents, sid, triples, revision_items)

    return result


def revise(sentences, triples):
    sent_info = '\n'.join(sentences)
    logger.info(f'Input sentences:\n{sent_info}')
    logger.info(f'Input triples: {triples}')
    result = []

    revisions = revise_sents(sentences, triples)
    for sid, sent in enumerate(sentences):
        sent = sentences[sid]
        revision_list = revisions.get(sid)
        if revision_list is None:
            # no revision is done: just output the original sentence text
            result.append({'original_text': sent,
                           'revised_text': sent,
                           'confidence_score': 1.0,
                           'confidence_label': 'High'})
        else:
            for revised_sent, rev_conf_score, rev_conf_label in revision_list:
                result.append({'original_text': sent,
                               'revised_text': revised_sent,
                               'confidence_score': rev_conf_score,
                               'confidence_label': rev_conf_label})

    return result

def alignment_label(a, b, round_id):

    start_symbol = "<S"+str(round_id)+">"
    end_symbol = "</S"+str(round_id)+">"

    a = a.lower().replace(' - ', '-').replace('[', '').replace(']', '').replace('.', '').replace(',', '').strip().split(' ') # lower for event
    b = b.lower().replace(' - ', '-').replace('[', '').replace(']', '').replace('.', '').replace(',', '').strip().split(' ') # lower for event

    temp = difflib.SequenceMatcher(None, a,b)
    cnt_insert = 0
    match_blocks = temp.get_matching_blocks()[:-1]

    for i, block in enumerate(match_blocks):

        match_aid = block[0]
        match_bid = block[1]
        match_size = block[2]

        # Processing the case of first element not matched
        if i == 0:
            if a[0] != b[0]:
                a.insert(0, start_symbol)
                b.insert(0, start_symbol)
                cnt_insert += 1

        # # insert split sign to a and b
        # print(match_aid+match_size+cnt_insert)
        a.insert(match_aid+match_size+cnt_insert, start_symbol)
        b.insert(match_bid+match_size+cnt_insert, start_symbol)
        cnt_insert+=1

    cnt_insert = 0
    temp = difflib.SequenceMatcher(None, a,b)
    match_blocks = temp.get_matching_blocks()[:-1]
    # print("Adding </S>", match_blocks)

    for i, block in enumerate(match_blocks):

        match_aid = block[0]
        match_bid = block[1]
        match_size = block[2]

        # Processing the case of first element not matched
        if i == 0:
            if a[0] != b[0]:
                a.insert(match_aid, end_symbol)
                b.insert(match_bid, end_symbol)
                cnt_insert+=1

        try:
            a_end_id, b_end_id, next_match_size = match_blocks[i+1]
            a.insert(a_end_id+cnt_insert, end_symbol)
            b.insert(b_end_id+cnt_insert, end_symbol)
        except:
            a.append(end_symbol)
            b.append(end_symbol)

        cnt_insert+=1

        # print(a)
        # print(b)

    a = ' '.join(a).replace('[', '').replace(']', '').replace(' '+start_symbol+' '+end_symbol+' ', ' ').replace(start_symbol+' '+end_symbol, '')
    b = ' '.join(b).replace('[', '').replace(']', '').replace(' '+start_symbol+' '+end_symbol+' ', ' ').replace(start_symbol+' '+end_symbol, '')

    return a, b

def gpt_corrupt(triple, ctype='entity', ents=None, rels=None):

    if ctype == "entity":
        triple_examples = [
            {
                'old_triple': ('Ben', 'works for', 'Fruit Company'),
                'revised_triple': ('Ben', 'works for', 'Apple'),
            },
            {
                'old_triple': ('Aarhus Airport', 'city Served', 'Aarhus Denmark'),
                'revised_triple': ('Aarhus Airport', 'city Served', 'New York'),
            },

        ]
    else:
        triple_examples = [
            {
                'old_triple': ('Aarhus Airport', 'operating Organisation', 'Aarhus Lufthavn A/S'),
                'revised_triple': ('Aarhus Airport', 'leader Name', 'Aarhus Lufthavn A/S'),
            },

            {
                'old_triple': ('Aarhus Airport', 'location', 'Tirstrup'),
                'revised_triple': ('Aarhus Airport', 'country', 'Tirstrup'),
            },

            {
                'old_triple': ('Aarhus Airport', 'location', 'Tirstrup'),
                'revised_triple': ('Aarhus Airport', 'birthday', 'Tirstrup'),
            },
            {
                'old_triple': ("jamaica at the fifa world cup", "subclass of", "jamaica national football team"),
                'revised_triple': ("jamaica at the fifa world cup", "president of", "jamaica national football team"),
            },
            {
                'old_triple': ("kentucky\u2013louisville rivalry", "participating team", "louisville cardinals"),
                'revised_triple': ("kentucky\u2013louisville rivalry", "beat", "louisville cardinals"),
            }
        ]

    triple_example_template = """
    Old triple: {old_triple}
    New triple: {revised_triple}
    """

    triple_example_prompt = PromptTemplate(
        input_variables=['old_triple', 'revised_triple'],
        template=triple_example_template,
    )

    if ctype == "entity":
        _revise_triple_prompt = \
            FewShotPromptTemplate(examples=triple_examples,
                                  example_prompt=triple_example_prompt,
                                  prefix="Using your commonsense knowledge to edit either the object or subject in the old triple to make it counterfactual.",
                                  suffix='Old triple: {old_triple}\nNew triple:',
                                  input_variables=['old_triple'],
                                  example_separator='\n')
    else:
        _revise_triple_prompt = \
            FewShotPromptTemplate(examples=triple_examples,
                                  example_prompt=triple_example_prompt,
                                  prefix="Using your commonsense knowledge to edit the predicate in the old triple to make it counterfactual. Note that you should not always use predicate negation.",
                                  suffix='Old triple: {old_triple}\nNew triple:',
                                  input_variables=['old_triple'],
                                  example_separator='\n')

    _revise_triple_chain = LLMChain(llm=llm_triple, prompt=_revise_triple_prompt)


    new_triple = _revise_triple_chain.run(
        old_triple=triple,
    )
    logger.info(f"[Revising Triple]\nold: {triple}\nnew: {new_triple}")

    try:
        new_triple = list(ast.literal_eval(new_triple.strip()))
    except:
        # Randomlly corrupt
        corrupt_type = random.choice([0, 1, 2]) # 0 for head, 1 for r, 2 for tail
        if corrupt_type == 0 or corrupt_type == 2:
            triple[corrupt_type] = random.choice(ents)
        else:
            triple[corrupt_type] = random.choice(rels)
        new_triple = triple

    return new_triple

if __name__ == '__main__':
    random.seed(44)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    _revise_tests = [
        {
            'sentences': [
                'Farima is a PhD student.',
                'Kun is a computer scientist.',
                'Ben is a computer scientist and he works for Fruit Company.'],
            'triples': [
                (
                    ('Farima', 'occupation', 'PhD student'), # original triple
                    0, # sentence id
                    None, # new triple (s, p, o)
                    1.0, # confidence score
                    'High' # confidence level
                ),
                (
                    ('Kun', 'occupation', 'computer scientist'), # original triple
                    1, # sentence id
                    None, # new triple (s,p,o)
                    1.0, # confidence score
                    'High' # confidence level
                ),
                (
                    ('Ben', 'occupation', 'computer scientist'),
                    2,
                    ('Ben', 'occupation', 'janitor'),
                    1.0,
                    'High'
                ),
                (
                    ('Ben', 'works for', 'Fruit Company'),
                    2,
                    ('Ben', 'employed by', 'Apple Inc'),
                    0.6,
                    'Medium'
                ),
            ]
        },
        {
            'sentences': [
                'Written by Garth Nix, Aenir, has the ISBN number of 0-439-17684-0.'],
            'triples': [
                (
                    ('Aenir', 'author', 'Garth Nix'), # original triple
                    0, # sentence id
                    None, # new triple (s, p, o)
                    1.0, # confidence score
                    'High' # confidence level
                ),
                (
                    ('Aenir', 'ISBN number', '0-439-17684-0'), # original triple
                    0, # sentence id
                    ('Aenir', 'ID number', '0-0827-8172-187'), # new triple (s,p,o)
                    1.0, # confidence score
                    'High' # confidence level
                ),
            ]
        },
    ]

    df = pd.read_csv(input_file, sep='\t')
    str_triples = [d+'<END>' for d in df['source'].values]
    sentences = df['target'].values

    h_expression = r'<H>(.*?)<R>'
    r_expression = r'<R>(.*?)<T>'
    t_expression = r'<T>(.*?)<H>'
    tEND_expression = r'(?s:.*)<T>(.*?)<END>'


    triples = [] # list of list of turples
    ents = []
    rels = []
    for triple_str in str_triples:

        instance = []

        head_ents = [d.strip() for d in re.findall(h_expression, triple_str)]
        tail_ents = [d.strip() for d in re.findall(t_expression, triple_str) + re.findall(tEND_expression, triple_str)]
        relation_pred = [d.strip() for d in re.findall(r_expression, triple_str)]

        ents += head_ents
        ents += tail_ents
        rels += relation_pred

        assert len(head_ents) == len(tail_ents)
        assert len(head_ents) == len(relation_pred)

        num_triple = len(tail_ents)

        extracted_triples = [[head_ents[i],relation_pred[i],tail_ents[i]] for i in range(num_triple)]
        triples.append(extracted_triples)

    ents = list(set(ents))
    rels = list(set(rels))

    # # debugging
    triples = triples[:10]
    sentences = sentences[:10]

    # First edit triples, prepare revise_text that need to be revised
    results = []
    for sample_idx, d in enumerate(triples):
        # print("Before corruption:", [tuple(d) for d in sample_triples])
        sample_triples = copy.deepcopy(d)
        sample_sentence = copy.deepcopy(sentences[sample_idx])

        corrupt_record = []
        revised_history = []

        # for each instance, edit two steps
        for round_id in range(1):

            # print(" *******   Round ", round_id, " original triples:", sample_triples)
            # print(" *******   Round ", round_id, " original SENTENCE:", sample_sentence)

            corrupt_triples = copy.deepcopy(sample_triples)

            corrupt_idx = [j for j in range(len(corrupt_triples))] # randomly select one triple to edit
            corrupt_i = random.choice(corrupt_idx)
            corrupt_record.append(corrupt_i)

            corrupt_type = random.choice([0, 1]) # 0 for entity, 1 for predicate

            # Corruption with GPT
            try:
                if corrupt_type == 0:
                    corrupt_triples[corrupt_i] = gpt_corrupt(corrupt_triples[corrupt_i], ctype='entity', ents=ents, rels=rels)
                else:
                    corrupt_triples[corrupt_i] = gpt_corrupt(corrupt_triples[corrupt_i], ctype='relation', ents=ents, rels=rels)
            except:
                time.sleep(15)
                if corrupt_type == 0:
                    corrupt_triples[corrupt_i] = gpt_corrupt(corrupt_triples[corrupt_i], ctype='entity', ents=ents, rels=rels)
                else:
                    corrupt_triples[corrupt_i] = gpt_corrupt(corrupt_triples[corrupt_i], ctype='relation', ents=ents, rels=rels)

            revise_input = {}
            revise_input['sentences'] = [sample_sentence]
            revise_triples = []
            for i_triple, t in enumerate(sample_triples):
                original_triple = tuple(t)
                counterfactual_triple = tuple(corrupt_triples[i_triple])
                revise_triples.append((original_triple, 0, counterfactual_triple, 1.0, 'High'))
            revise_input['triples'] = revise_triples

            # Edit text and do alignment
            try:
                result = revise(**revise_input)[0]
            except:
                time.sleep(15)
                result = revise(**revise_input)[0]
            corrupt_sentence = result['revised_text']

            # Write into outpus
            result['corrupt_triple_idx'] = corrupt_record
            result['original_triple'] = [tuple(d) for d in triples[sample_idx]]
            result['counterfactual_triple'] = [tuple(d) for d in corrupt_triples]

            # find alignment using diff
            # print("ALIGNING: ", sample_sentence, corrupt_sentence)
            aligned_original_sent, aligned_corrupt_sent = alignment_label(sample_sentence, corrupt_sentence, round_id=round_id)
            result['original_text'] = aligned_original_sent
            result['revised_text'] = aligned_corrupt_sent
            revised_history.append(aligned_corrupt_sent)
            # print("ALIGNING Output: ", aligned_original_sent, aligned_corrupt_sent)

            sample_triples = corrupt_triples
            sample_sentence = corrupt_sentence

            # print(" *******   Round ", round_id, " corrupt triples:", corrupt_triples)
            # print(" *******   Round ", round_id, " corrupt SENTENCE:", corrupt_sentence)

        result['revised_history'] = revised_history

        results.append(result)

        with open(output_file, 'w') as f:
            f.write(json.dumps(results, indent=4))