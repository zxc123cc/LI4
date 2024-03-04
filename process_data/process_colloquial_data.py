import json

LABELS = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

def process_data(file_path,out_path):
    id = 0
    samples_dict = {}
    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            data = json.loads(line)
            evidence = data['evidences'][0]
            evidence_text = 'title: ' + evidence['title'] + ' content: ' + evidence['evidence']
            question = data['question']
            for claim in data['colloquial_claims']:
                sample = {}
                sample['id'] = id
                sample['claim'] = claim
                sample['evidence'] = evidence_text
                sample['question'] = question
                sample['label'] = data['fever_label']
                sample['label_id'] = LABELS[data['fever_label']]
                samples_dict[id] = sample
                id += 1

    print(len(samples_dict))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    process_data(file_path='../data/raw_data/colloquial/colloquial_claims_train_t5.jsonl', out_path='../data/tmp_data/colloquial/train.json')
    process_data(file_path='../data/raw_data/colloquial/colloquial_claims_valid_t5.jsonl', out_path='../data/tmp_data/colloquial/dev.json')
    process_data(file_path='../data/raw_data/colloquial/colloquial_claims_test_t5.jsonl', out_path='../data/tmp_data/colloquial/test.json')

