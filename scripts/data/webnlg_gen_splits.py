import os
import sys
from bs4 import BeautifulSoup
from unidecode import unidecode
import re

def camelcase_to_lowercase(string):
    result = string[0]
    for idx, char in enumerate(string[1:]):
        if char.isupper() and (string[idx].islower()):
            result += ' ' + char
        else:
            result += char
    return result

def get_triple(instance):
    triple_str = []
    for triple in instance.findAll("mtriple"):
        t_tok = triple.get_text().split("|")
        t = [tok.strip() for tok in t_tok]
        triple_str.append("<H> " + str(t[0]) + " <R> " + camelcase_to_lowercase(str(t[1])) + " <T> " + str(t[2]))
    triple_str = re.sub(r'[,()"]', '', " ".join(triple_str).replace("_", " "))
    triple_str = re.sub(r'\s+', ' ', triple_str)
    return unidecode(triple_str)

def get_text(instance):
    text = []
    for t in instance.findAll("lex"):
        t_text = t.get_text().replace('"', '""')
        if '"' in t_text:
            t_text = '"' + t_text + '"'
        text.append(t_text)
    return text

def main():

    dataset_directory = sys.argv[1]
    split_directory = sys.argv[2]
    validation_instance_file = sys.argv[3]
    test_instance_file = sys.argv[4]

    train_directory = os.path.join(dataset_directory, "webnlg_challenge_2017", "train")

    train_instances = []
    for i in range(1,8):
        dir = os.path.join(train_directory, str(i)+"triples")
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            with open(file_path, 'r') as f:
                data = f.read()
                xml_data = BeautifulSoup(data, "xml")
                entities = xml_data.findAll("entry")
                for instance in entities:
                    triples = get_triple(instance)
                    text = get_text(instance)
                    for t in text:
                        train_instances.append((triples, t))

    with open(os.path.join(split_directory, "train.csv"), "w") as f:
        f.write("source\ttarget\n")
        for t in train_instances:
            f.write("\t".join(t) + "\n")

    validation_directory = os.path.join(dataset_directory, "webnlg_challenge_2017", "dev")

    split_instances = set()
    with open(validation_instance_file, "r") as f:
        for line in f:
            split_instances.add(int(line))

    instances = []
    for i in range(1,8):
        dir = os.path.join(validation_directory, str(i)+"triples")
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            with open(file_path, 'r') as f:
                data = f.read()
                xml_data = BeautifulSoup(data, "xml")
                entities = xml_data.findAll("entry")
                for instance in entities:
                    triples = get_triple(instance)
                    text = get_text(instance)
                    for t in text:
                        instances.append((triples, t))

    with open(os.path.join(split_directory, "validation.csv"), "w") as f:
        f.write("source\ttarget\n")
        for idx, t in enumerate(instances):
            if idx in split_instances:
                f.write("\t".join(t) + "\n")

    test_dir = os.path.join(dataset_directory, "webnlg_challenge_2017", "test")

    split_instances = set()
    with open(test_instance_file, "r") as f:
        for line in f:
            split_instances.add(int(line))

    instances = []
    for file in ["testdata_unseen_with_lex.xml", "testdata_with_lex.xml"]:
        file_path = os.path.join(test_dir, file)
        with open(file_path, 'r') as f:
            data = f.read()
            xml_data = BeautifulSoup(data, "xml")
            entities = xml_data.findAll("entry")
            for instance in entities:
                triples = get_triple(instance)
                text = get_text(instance)
                for t in text:
                    instances.append((triples, t))

    with open(os.path.join(split_directory, "test.csv"), "w") as f:
        f.write("source\ttarget\n")
        for idx, t in enumerate(instances):
            if idx in split_instances:
                f.write("\t".join(t) + "\n")
main()