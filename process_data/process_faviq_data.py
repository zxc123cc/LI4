import json

LABELS = {"SUPPORTS":0, "REFUTES":1}

def process_data(file_path,out_path):
    num_na = 0
    data_dict = {}
    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data['label_id'] = LABELS[data['label']]
            data_dict[data['id']] = data
            if data['positive_evidence']['id'] == "N/A":
                num_na += 1

    print('Total: ',len(data_dict),' Not N/A: ',len(data_dict)-num_na, ' N/A: ',num_na, ' N/A proportion： ',num_na/len(data_dict))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def process_data_dpr(file_path,drp_path,out_path=None,max_evidences=3):
    raw_data_list = []
    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data['label_id'] = LABELS[data['label']]
            raw_data_list.append(data)
    data_dict = {}
    num = 0
    dpr_data_list = json.load(open(drp_path,'r',encoding='utf-8'))
    for raw_data,dpr_data in zip(raw_data_list[:],dpr_data_list[:]):
        if raw_data['claim'] != dpr_data['question']:
            num += 1
            print('claim != dpr question')
        dpr_evidences = dpr_data['ctxs']
        raw_data['dpr_evidences'] = []
        for e in dpr_evidences:
            if e['has_answer'] == True and e['id'] != 'N/A':
                raw_data['dpr_evidences'].append(e)
                if len(raw_data['dpr_evidences']) >= max_evidences:
                    break
        for e in dpr_evidences:
            if len(raw_data['dpr_evidences']) >= max_evidences:
                break
            if e not in raw_data['dpr_evidences'] and e['id'] != 'N/A':
                raw_data['dpr_evidences'].append(e)

        # raw_data['dpr_evidences'] = dpr_evidences[:max_evidences]

        # for e in dpr_evidences:
        #     if e['id'] != 'N/A':
        #         raw_data['dpr_evidences'].append(e)
        #     if len(raw_data['dpr_evidences']) >= max_evidences:
        #         break

        data_dict[raw_data['id']] = raw_data
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    process_data(file_path='../data/raw_data/faviq_a_set/train.jsonl', out_path='../data/tmp_data/faviq_a_set/train.json')
    process_data(file_path='../data/raw_data/faviq_a_set/dev.jsonl', out_path='../data/tmp_data/faviq_a_set/dev.json')

    process_data(file_path='../data/raw_data/faviq_r_set/train.jsonl', out_path='../data/tmp_data/faviq_r_set/train.json')
    process_data(file_path='../data/raw_data/faviq_r_set/dev.jsonl', out_path='../data/tmp_data/faviq_r_set/dev.json')
    process_data(file_path='../data/raw_data/faviq_r_set/test.jsonl', out_path='../data/tmp_data/faviq_r_set/test.json')

    process_data_dpr(file_path='../data/raw_data/faviq_a_set/train.jsonl',
                     drp_path='../data/raw_data/dpr_faviq/faviq_train_w_evidentiality.json',
                     out_path='../data/tmp_data/dpr_faviq/train3.json')
    process_data_dpr(file_path='../data/raw_data/faviq_a_set/dev.jsonl',
                     drp_path='../data/raw_data/dpr_faviq/faviq_dev.json',
                     out_path='../data/tmp_data/dpr_faviq/dev3.json')
    # process_data_dpr(file_path='./data/raw_data/faviq_a_set/test.jsonl',
    #                  drp_path='./data/raw_data/dpr_faviq/faviq_test.json',
    #                  out_path='./data/tmp_data/dpr_faviq/test.json')
