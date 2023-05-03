import json

import pandas as pd


def to_jsonl(json_lines: list, filename: str):
    with open(f"output/{filename}.jsonl", 'w') as outfile:
        for line in json_lines:
            json.dump(line, outfile)
            outfile.write('\n')


def to_csv(filename: str):
    filepath = f"output/{filename}"
    df = pd.read_json(filepath + ".jsonl", lines=True)
    df.to_csv(filepath + ".csv")


def write_jsons(activities_jsonl, constraints_jsonl, file, relations_jsonl):
    to_jsonl(constraints_jsonl, filename=file + "_constraints")
    to_jsonl(activities_jsonl, filename=file + "_activities")
    to_jsonl(relations_jsonl, filename=file + "_relations")


def get_data(data_path, data_type):
    if data_type == "jsonl":
        df = pd.read_json(data_path, lines=True)
    elif data_type == "json":
        df = pd.read_json(data_path)
    else:
        df = pd.read_csv(data_path)
    return df