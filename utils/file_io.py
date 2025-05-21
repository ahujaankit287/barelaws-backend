import json

def read_jsonl(filepath):
    """
    Reads a .jsonl file and returns a list of JSON objects (dicts).
    Skips empty lines automatically.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(filepath, data):
    """
    Writes a list of JSON objects (dicts) to a .jsonl file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')