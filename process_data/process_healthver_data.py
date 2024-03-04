import json
import pandas as pd

label_map = {"Supports":0, "Refutes":1, "Neutral":2}

def read_csv(input_file):
    data = pd.read_csv(input_file)
    lines = []
    for i in range(len(data)):
        lines.append(data.iloc[i])
    return lines

def convert_data_csv2json(csv_path,json_path,mode = 'train',remove_repeat=False):
    lines = read_csv(csv_path)
    sample_dict = {}
    if remove_repeat:
        tmp_dict = {}
    for line in lines[:]:
        if remove_repeat:
            now_tuple = (line['question'],line['evidence'],line['claim'])
            if now_tuple not in tmp_dict:
                tmp_dict[now_tuple] = 1
            else:
                print(now_tuple)
                continue
        id = int(line['id'])
        sample_dict[id] = {
            'id' : id,
            'topic_ip' : int(line['topic_ip']),
            'question' : line['question'],
            'evidence' : line['evidence'],
            'claim' : line['claim'],
        }
        if mode == 'train' or mode == 'dev':
            sample_dict[id]['label'] = line['label']
            sample_dict[id]['label_id'] = label_map[line['label']]

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    convert_data_csv2json(csv_path='../data/raw_data/healthver/healthver_train.csv',
                          json_path='../data/tmp_data/healthver/healthver_train.json', mode='train')
    convert_data_csv2json(csv_path='../data/raw_data/healthver/healthver_dev.csv',
                          json_path='../data/tmp_data/healthver/healthver_dev.json', mode='dev')
    convert_data_csv2json(csv_path='../data/raw_data/healthver/healthver_test.csv',
                          json_path='../data/tmp_data/healthver/healthver_test_with_label.json', mode='dev')