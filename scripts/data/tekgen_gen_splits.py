import os
import json
import sys

def get_triple(instance):
    triple_str = []
    for t in instance["triples"]:
        triple_str.append("<H> " + str(t[0]) + " <R> " + str(t[1]) + " <T> " + str(t[2]))
    triple_str = " ".join(triple_str)
    return triple_str

def get_text(instance):
    sentence = instance["sentence"].replace('"', '""')
    if '"' in sentence:
        sentence = '"' + sentence + '"'
    return sentence

def main():

    dataset_directory = sys.argv[1]
    split_directory = sys.argv[2]
    train_instance_file = sys.argv[3]
    validation_instance_file = sys.argv[4]
    test_instance_file = sys.argv[5]

    train_file = os.path.join(dataset_directory, "quadruples-train.tsv")
    validation_file = os.path.join(dataset_directory, "quadruples-validation.tsv")
    test_file = os.path.join(dataset_directory, "quadruples-test.tsv")


    # Create train split
    all_instance = []
    split_instances = set()
    with open(train_file, "r") as f:
        for line in f:
            all_instance.append(json.loads(line))

    with open(train_instance_file, "r") as f:
        for line in f:
            split_instances.add(int(line))

    with open(os.path.join(split_directory, "tekgen-train.csv"), "w") as f:
        f.write("source\ttarget\n")
        for idx, i in enumerate(all_instance):
            if idx in split_instances:
                f.write(get_triple(i) + "\t" + get_text(i) + "\n")

    # Create validation split
    all_instance = []
    split_instances = set()
    with open(validation_file, "r") as f:
        for line in f:
            all_instance.append(json.loads(line))

    with open(validation_instance_file, "r") as f:
        for line in f:
            split_instances.add(int(line))

    with open(os.path.join(split_directory, "tekgen-validation.csv"), "w") as f:
        f.write("source\ttarget\n")
        for idx, i in enumerate(all_instance):
            if idx in split_instances:
                f.write(get_triple(i) + "\t" + get_text(i) + "\n")

    # Create test split
    all_instance = []
    split_instances = set()
    with open(test_file, "r") as f:
        for line in f:
            all_instance.append(json.loads(line))

    with open(test_instance_file, "r") as f:
        for line in f:
            split_instances.add(int(line))

    with open(os.path.join(split_directory, "tekgen-test.csv"), "w") as f:
        f.write("source\ttarget\n")
        for idx, i in enumerate(all_instance):
            if idx in split_instances:
                f.write(get_triple(i) + "\t" + get_text(i) + "\n")

main()