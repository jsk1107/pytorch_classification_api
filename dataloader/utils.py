def label_map(file_path):
    classes = {}
    with open(file_path, 'r', encoding='utf-8') as t:
        lines = t.read().splitlines()
        for i, line in enumerate(lines):
            classes[line] = i
    return classes


# def dacon_preprocess():
