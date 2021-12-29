import pandas as pd
import csv
import json
import re


def place_numbers(text, numbers):
    for i, num in enumerate(numbers.split()):
        # text = text.replace(f'number{i}', f'NUM_{i}')
        text = re.sub(f'[Nn]umber{i}', f'NUM_{i}', text)
        # text = text.replace(f'number{i}', num)
    return text


def change_op(text):
    text = re.sub(r'\+', 'add', text)
    text = re.sub(r'\-', 'sub', text)
    text = re.sub(r'\*', 'mul', text)
    text = re.sub(r'\/', 'div', text)
    return text


def main():
#     src_file_names = ['train.csv', 'dev.csv', 'dev.csv']
#     tgt_file_names = ['trainset.json', 'validset.json', 'testset.json']
    src_file_names = ['train.csv', 'dev.csv']
    tgt_file_names = ['validset.json', 'testset.json']

    for src, tgt in zip(src_file_names, tgt_file_names):
        c = pd.read_csv(src, encoding='utf-8').reset_index().rename(columns={'index': 'ID'})
        for col in ['Question', 'Equation', 'Body', 'Ques_Statement']:
            c[col] = c.apply(lambda row: place_numbers(row[col], row['Numbers']), axis=1)
        c['Equation'] = c['Equation'].apply(lambda x: change_op(x))
        d = c.to_json(None, orient='records', force_ascii=False, indent=4)
        with open(tgt, 'w', encoding='utf-8') as f:
            f.write(d)


if __name__ == '__main__':
    main()
