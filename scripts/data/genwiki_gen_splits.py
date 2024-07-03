import os
import json
import sys

def get_triple(instance):
    triple_str = []
    for t in instance["graph"]:
        triple_str.append("<H> " + str(t[0]) + " <R> " + str(t[1]) + " <T> " + str(t[2]))
    triple_str = " ".join(triple_str).replace('"', '""')
    if '"' in triple_str:
        triple_str = '"' + triple_str + '"'
    return triple_str

def get_text(instance):
    entities = {}
    i = 0
    for e in instance["entities"]:
        entities["<ENT_" + str(i) + ">"] = e
        i += 1

    text = instance["text"]
    for e in entities:
        text = text.replace(e,entities[e])

    return text

def main():

    dataset_directory = sys.argv[1]
    split_directory = sys.argv[2]
    validation_lines_file = sys.argv[3]

    validation_lines = []
    with open(validation_lines_file, "r") as f:
        for line in f:
            validation_lines.append(int(line))

    train_directory = os.path.join(dataset_directory, "train", "fine")
    test_directory = os.path.join(dataset_directory, "test")

    instance = []
    train_instances = []
    validation_instances = []

    # The training split contains 76 files
    for i in range(1,77):
        f = os.path.join(train_directory, "part_" + str(i) + ".json")
        if os.path.isfile(f):
            with open(f, "r") as f:
                data = json.load(f)
                instance.extend(data)

    for line, i in enumerate(instance):
        triple = get_triple(i)
        text = get_text(i)
        if line in validation_lines:
            validation_instances.append((triple, text))
        else:
            train_instances.append((triple, text))

    with open(os.path.join(split_directory, "genwiki-train.csv"), "w") as f:
        f.write("source\ttarget\n")
        for instance in train_instances:
            f.write("\t".join(instance) + "\n")

    with open(os.path.join(split_directory, "genwiki-validation.csv"), "w") as f:
        f.write("source\ttarget\n")
        for instance in validation_instances:
            f.write("\t".join(instance) + "\n")

    test_instances = []
    f = os.path.join(test_directory, "test.json")
    if os.path.isfile(f):
        with open(f, "r") as f:
            data = json.load(f)
            for instance in data:
                test_instances.append((get_triple(instance), get_text(instance)))

    with open(os.path.join(split_directory, "genwiki-test.csv"), "w") as f:
        f.write("source\ttarget\n")
        for instance in test_instances:
            f.write("\t".join(instance) + "\n")

main()