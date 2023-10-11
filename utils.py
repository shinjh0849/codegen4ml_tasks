import csv
import json
import math
import os
from pathlib import Path
import random
# import matplotlib.pyplot as plt
import numpy as np
import keywords
import pickle

#juice split: dev: 47%, test: 53%\

def rm_dups_from_train(indir):
    for ds in ['combined', 'data', 'eval', 'model', 'nonml']:
        train_list = []
        test_list = []
        dev_list = []

        new_train = []
        new_test = []

        train_path = '{}/train/{}.json'.format(indir, ds)
        with open(train_path) as read_file:
            train_list = read_file.readlines()
                

        test_path = '{}/test/{}.json'.format(indir, ds)
        with open(test_path) as read_file:
            test_list = read_file.readlines()

        dev_path = '{}/dev/{}.json'.format(indir, ds)
        with open(dev_path) as read_file:
            dev_list = read_file.readlines()

        for test in test_list:
            if test not in dev_list:
                new_test.append(test)

        for train in train_list:
            if train not in new_test and train not in dev_list:
                new_train.append(train)


        Path('{}/new_test/'.format(indir)).mkdir(parents=True, exist_ok=True)
        Path('{}/new_train/'.format(indir)).mkdir(parents=True, exist_ok=True)

        with open('{}/new_train/{}.json'.format(indir, ds), 'w') as train_file:
            for line in new_train:
                train_file.write(line)
        with open('{}/new_test/{}.json'.format(indir, ds), 'w') as test_file:
            for line in new_test:
                test_file.write(line)
            
    return

def count_nl():
    data_split = 'train'

    # traverse all files in the directory
    directory = 'rm_dups/train/'
    loc = []
    length = []
    for filename in os.listdir(directory):
        read_file = os.path.join(directory, filename)
        print(read_file)
        input_file = open(read_file, 'r')
        lines = input_file.readlines()
        for line in lines:
            dict_line = json.loads(line)
            loc.append(len(dict_line['snippet'].splitlines()))
            length.append(len(dict_line['intent'].split))
        print('loc:', sum(loc) / len(loc))
        print('length:', sum(length) / len(length))

    input_file.close()

def add_comma(train, test):
    train_path = 'new_filtered/' + train
    test_path = 'new_filtered/' + test

    new_train = open(train, 'w')
    new_test = open(test, 'w')

    train_f = open(train_path, 'r')
    test_f = open(test_path, 'r')

    train_lines = train_f.readlines()
    test_lines = test_f.readlines()

    for train_line in train_lines:
        new_train.write(train_line.strip() + ',\n')

    for test_line in test_lines:
        new_test.write(test_line.strip() + ',\n')

    new_train.close()
    new_test.close()
    train_f.close()
    test_f.close()

def jsonlines_to_json():
    add_comma('train.json', 'test.json')

    filepath = 'new_filtered/train.json'
    juice_file = open(filepath, 'rb')
    juice_lines = juice_file.readlines()
    src_f = open('src.en','w')
    write_json = open('new_juice_train.json', 'w', encoding='utf-8')

    for i, data in enumerate(juice_lines):
        d = json.loads(data)
        nl = d['nl']
        code = d['code']
        nl_strip = nl.replace("\n", "")
        code_strip = code.replace("\n", "")

        try:
            code_strip.encode().decode('ascii')
            nl_strip.encode().decode('ascii')
            src_f.write(nl_strip + '\n')
            json.dump(d, write_json)
            write_json.write('\n')
        except UnicodeError:
            print(str(i) + ' is not utf-8')
    juice_file.close()
    src_f.close()
    write_json.close()

def count_frequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
    
    # for key, value in freq.items():
    #     print ("% d : % d"%(key, value))

def get_loc_dist(in_dir):
    open_file = open(in_dir, 'r')
    lines = open_file.readlines()
    code_loc_list = []
    nl_len_list = []
    api_num_list = []
    for i, data in enumerate(lines):
        d = json.loads(data)
        code = d['snippet']
        nl = d['intent']
        api_num = 0
        for data_kw in keywords.data_process:
            if data_kw in code:
                api_num += 1
        for model_kw in keywords.model_building:
            if model_kw in code:
                api_num += 1
        for eval_kw in keywords.evaluation_visual:
            if eval_kw in code:
                api_num += 1
        
        api_num_list.append(api_num) 
        code_loc = len(code.splitlines())
        nl_len = len(nl.split())
        code_loc_list.append(code_loc)
        nl_len_list.append(nl_len)
    print('code_loc:')
    count_frequency(code_loc_list)
    print('average:', sum(code_loc_list) / len(code_loc_list))
    print('quartiles', np.quantile(code_loc_list, [0, 0.25, 0.5, 0.75, 1]))

    print('nl_len:')
    count_frequency(nl_len_list)
    print('average:', sum(nl_len_list) / len(nl_len_list))
    print('quartiles', np.quantile(nl_len_list, [0, 0.25, 0.5, 0.75, 1]))

    print('api number:')
    print('average:', sum(api_num_list) / len(api_num_list))

    # plt.hist(loc_list, bins=250) 
    # plt.axis([0, 20, 0, 160000]) 
    # #axis([xmin,xmax,ymin,ymax])

    # plt.show()
    open_file.close()

# my random function
def my_random():
    return 0.1

def split(original_list, weight_list):
    sublists = []
    prev_index = 0
    for weight in weight_list:
        next_index = prev_index + math.ceil((len(original_list) * weight))

        sublists.append(original_list[prev_index : next_index])
        prev_index = next_index

    return sublists

def rm_dups(dup_lines):
    seen = set()
    answer = []
    for line in dup_lines:
        if line not in seen:
            seen.add(line)
            answer.append(line)
            
    return ''.join(answer)

def make_dev_n_test_set(input_dir, is_non_ml=False):
    Path(input_dir + '/test/').mkdir(parents=True, exist_ok=True)
    Path(input_dir + '/dev/').mkdir(parents=True, exist_ok=True)
    if is_non_ml:
        ds_list = ['non_ml']
    else:
        ds_list = ['data', 'model', 'eval', 'combined']
    csvfile_dev = open(input_dir + '/dev_stats.csv', 'w')
    csvfile_test = open(input_dir + '/test_stats.csv', 'w')
    csvwriter_dev = csv.writer(csvfile_dev)
    csvwriter_test = csv.writer(csvfile_test)

    for ds in ds_list:
        combined = []
        dev_row = []
        test_row = []

        # remove duplicates from train
        train_file = open(input_dir + '/train/' + ds + '.json', 'r')
        train_lines = train_file.readlines()
        train_lines = rm_dups(train_lines)
        train_file.close()
        train_file = open(input_dir + '/train/' + ds + '2.json', 'w')
        for line in train_lines.splitlines():
            train_file.write(line+'\n')
        train_file.close()

        nbg_file = open(input_dir + '/nbgrader/' + ds + '.json', 'r')
        nbg_lines = nbg_file.readlines()
        nbg_lines = rm_dups(nbg_lines)
        nbg_file.close()

        for data in nbg_lines.splitlines():
            combined.append(data)

        exc_file = open(input_dir + '/exercise/' + ds + '.json', 'r')
        exc_lines = exc_file.readlines()
        exc_lines = rm_dups(exc_lines)
        exc_file.close()
        
        for data in exc_lines.splitlines():
            combined.append(data)

        random.shuffle(combined, my_random)
        
        dev_file = open(input_dir + '/dev/' + ds + '.json', 'w')
        test_file = open(input_dir + '/test/' + ds + '.json', 'w')

        split_list = split(combined, [0.1, 0.9])

        for line in split_list[0]:
            dev_file.write(line+'\n')
        dev_file.close()

        for line in split_list[1]:
            test_file.write(line+'\n')
        test_file.close()

        dev_row.append(ds)
        dev_row.append(len(split_list[0]))
        test_row.append(ds)
        test_row.append(len(split_list[1]))

        csvwriter_dev.writerow(dev_row)
        csvwriter_test.writerow(test_row)
    
    csvfile_dev.close()
    csvfile_test.close()
    return

def get_samples_nonml(in_dir):
    train = open(in_dir + '/train/non_ml.json', 'r')
    dev = open(in_dir + '/dev/non_ml.json', 'r')
    test = open(in_dir + '/test/non_ml.json', 'r')

    train_lines = train.readlines()
    dev_lines = dev.readlines()
    test_lines = test.readlines()

    random.shuffle(train_lines, my_random)
    random.shuffle(dev_lines, my_random)
    random.shuffle(test_lines, my_random)


    train_num = 35028
    dev_num = 616
    test_num = 5535


    if len(train_lines) < train_num:
        print('train less than num')
    else: 
        train_lines = train_lines[:train_num]
    if len(dev_lines) < dev_num:
        print('dev less than num')
    else:
        dev_lines = dev_lines[:dev_num]
    if len(test_lines) < test_num:
        print('test less than num')
    else:
        test_lines = test_lines[:test_num]

    train_out = open(in_dir + '/train_nonml.json', 'w')
    for row in train_lines:
        d = json.loads(row)
        json.dump(d, train_out)
        train_out.write('\n')
    train_out.close()

    dev_out = open(in_dir + '/dev_nonml.json', 'w')
    for row in dev_lines:
        d = json.loads(row)
        json.dump(d, dev_out)
        dev_out.write('\n')
    dev_out.close()

    test_out = open(in_dir + '/test_nonml.json', 'w')
    for row in test_lines:
        d = json.loads(row)
        json.dump(d, test_out)
        test_out.write('\n')
    test_out.close()

    return

def get_all_random_samples(in_dir):
    train_model = open(in_dir + '/train/model.json', 'r')
    # train_feature = open(in_dir + '/train/feature.json', 'r')
    train_data = open(in_dir + '/train/data.json', 'r')
    train_eval = open(in_dir + '/train/eval.json', 'r')
    train_comb = open(in_dir + '/train/combined.json', 'r')
    train_model_lines = train_model.readlines()
    # train_feature_lines = train_feature.readlines()
    train_data_lines = train_data.readlines()
    train_eval_lines = train_eval.readlines()
    train_comb_lines = train_comb.readlines()

    test_model = open(in_dir + '/test/model.json', 'r')
    # test_feature = open(in_dir + '/test/feature.json', 'r')
    test_data = open(in_dir + '/test/data.json', 'r')
    test_eval = open(in_dir + '/test/eval.json', 'r')
    test_comb = open(in_dir + '/test/combined.json', 'r')
    test_model_lines = test_model.readlines()
    # test_feature_lines = test_feature.readlines()
    test_data_lines = test_data.readlines()
    test_eval_lines = test_eval.readlines()
    test_comb_lines = test_comb.readlines()

    dev_model = open(in_dir + '/dev/model.json', 'r')
    # dev_feature = open(in_dir + '/dev/feature.json', 'r')
    dev_data = open(in_dir + '/dev/data.json', 'r')
    dev_eval = open(in_dir + '/dev/eval.json', 'r')
    dev_comb = open(in_dir + '/dev/combined.json', 'r')
    dev_model_lines = dev_model.readlines()
    # dev_feature_lines = dev_feature.readlines()
    dev_data_lines = dev_data.readlines()
    dev_eval_lines = dev_eval.readlines()
    dev_comb_lines = dev_comb.readlines()

    all_model = dev_model_lines + test_model_lines + train_model_lines
    # all_feature = dev_feature_lines + test_feature_lines + train_feature_lines
    all_data = dev_data_lines + test_data_lines + train_data_lines
    all_eval = dev_eval_lines + test_eval_lines + train_eval_lines
    all_comb = dev_comb_lines + test_comb_lines + train_comb_lines

    random.shuffle(all_model, my_random)
    # random.shuffle(all_feature, my_random)
    random.shuffle(all_data, my_random)
    random.shuffle(all_eval, my_random)
    random.shuffle(all_comb, my_random)

    all_data = all_data[:100]
    # all_feature = all_feature[:100]
    all_model = all_model[:100]
    all_eval = all_eval[:100]
    all_comb = all_comb[:100]

    model_out = open(in_dir + '/model_sample.csv', 'w')
    model_csv = csv.writer(model_out)
    model_csv.writerow(['NL description', 'Source Code'])
    for row in all_model:
        d = json.loads(row)
        nl = d['nl']
        sc = d['code']
        model_csv.writerow([nl, sc])
    model_out.close()

    # feature_out = open(in_dir + '/feature_sample.csv', 'w')
    # feature_csv = csv.writer(feature_out)
    # feature_csv.writerow(['NL description', 'Source Code'])
    # for row in all_feature:
    #     d = json.loads(row)
    #     nl = d['nl']
    #     sc = d['code']
    #     feature_csv.writerow([nl, sc])
    # feature_out.close()

    data_out = open(in_dir + '/data_sample.csv', 'w')
    data_csv = csv.writer(data_out)
    data_csv.writerow(['NL description', 'Source Code'])
    for row in all_data:
        d = json.loads(row)
        nl = d['nl']
        sc = d['code']
        data_csv.writerow([nl, sc])
    data_out.close()

    eval_out = open(in_dir + '/eval_sample.csv', 'w')
    eval_csv = csv.writer(eval_out)
    eval_csv.writerow(['NL description', 'Source Code'])
    for row in all_eval:
        d = json.loads(row)
        nl = d['nl']
        sc = d['code']
        eval_csv.writerow([nl, sc])
    eval_out.close()

    comb_out = open(in_dir + '/combined_sample.csv', 'w')
    comb_csv = csv.writer(comb_out)
    comb_csv.writerow(['NL description', 'Source Code'])
    for row in all_comb:
        d = json.loads(row)
        nl = d['nl']
        sc = d['code']
        comb_csv.writerow([nl, sc])
    comb_out.close()

    return

def get_cnt_of_libs(in_dir):
    
    sklearn_cnt = 0
    tf_cnt = 0
    keras_cnt = 0
    torch_cnt = 0
    non_cnt = 0
    total_cnt = 0

    ds_splits=['combined', 'data', 'eval', 'model']

    # iterate all ml-tasks
    for ds in ds_splits:

        train = open(in_dir + '/train/'+ds+'.json', 'r')
        dev = open(in_dir + '/dev/'+ds+'.json', 'r')
        test = open(in_dir + '/test/'+ds+'.json', 'r')
        train_lines = train.readlines()
        dev_lines = dev.readlines()
        test_lines = test.readlines()

        # iterate all train instances
        for row in train_lines:
            total_cnt += 1
            kw_flag = False
            d = json.loads(row)
            code = d['snippet']
            desc = d['intent']

            # tensorflow
            if any(tf_keyword.casefold() in code.casefold() for tf_keyword in keywords.tensorflow_keywords) or \
               any(tf_keyword.casefold() in desc.casefold() for tf_keyword in keywords.tensorflow_keywords):
                tf_cnt += 1
                kw_flag = True
            # keras
            if any(kr_keyword.casefold() in code.casefold() for kr_keyword in keywords.keras_keywords) or \
                any(kr_keyword.casefold() in desc.casefold() for kr_keyword in keywords.keras_keywords):
                keras_cnt += 1
                kw_flag = True
            # pytorch
            if any(pt_word.casefold() in code.casefold() for pt_word in keywords.pytorch_keywords) or \
               any(pt_word.casefold() in desc.casefold() for pt_word in keywords.pytorch_keywords): 
                torch_cnt += 1
                kw_flag = True
            # sklearn
            if any(sk_keyword.casefold() in code.casefold() for sk_keyword in keywords.sklearn_keywords) or \
               any(sk_keyword.casefold() in desc.casefold() for sk_keyword in keywords.sklearn_keywords):
                sklearn_cnt += 1
                kw_flag = True
            if not kw_flag:
                non_cnt += 1

        # iterate all dev_instances
        for row in dev_lines:
            total_cnt += 1
            kw_flag = False
            d = json.loads(row)
            code = d['snippet']
            desc = d['intent']

            # tensorflow
            if any(tf_keyword.casefold() in code.casefold() for tf_keyword in keywords.tensorflow_keywords) or \
               any(tf_keyword.casefold() in desc.casefold() for tf_keyword in keywords.tensorflow_keywords):
                tf_cnt += 1
                kw_flag = True
            # keras
            if any(kr_keyword.casefold() in code.casefold() for kr_keyword in keywords.keras_keywords) or \
                any(kr_keyword.casefold() in desc.casefold() for kr_keyword in keywords.keras_keywords):
                keras_cnt += 1
                kw_flag = True
            # pytorch
            if any(pt_word.casefold() in code.casefold() for pt_word in keywords.pytorch_keywords) or \
               any(pt_word.casefold() in desc.casefold() for pt_word in keywords.pytorch_keywords): 
                torch_cnt += 1
                kw_flag = True
            # sklearn
            if any(sk_keyword.casefold() in code.casefold() for sk_keyword in keywords.sklearn_keywords) or \
               any(sk_keyword.casefold() in desc.casefold() for sk_keyword in keywords.sklearn_keywords):
                sklearn_cnt += 1
                kw_flag = True
            if not kw_flag:
                non_cnt += 1

        # iterate all test instances
        for row in test_lines:
            total_cnt += 1
            kw_flag = False
            d = json.loads(row)
            code = d['snippet']
            desc = d['intent']

            # tensorflow
            if any(tf_keyword.casefold() in code.casefold() for tf_keyword in keywords.tensorflow_keywords) or \
               any(tf_keyword.casefold() in desc.casefold() for tf_keyword in keywords.tensorflow_keywords):
                tf_cnt += 1
                kw_flag = True
            # keras
            if any(kr_keyword.casefold() in code.casefold() for kr_keyword in keywords.keras_keywords) or \
                any(kr_keyword.casefold() in desc.casefold() for kr_keyword in keywords.keras_keywords):
                keras_cnt += 1
                kw_flag = True
            # pytorch
            if any(pt_word.casefold() in code.casefold() for pt_word in keywords.pytorch_keywords) or \
               any(pt_word.casefold() in desc.casefold() for pt_word in keywords.pytorch_keywords): 
                torch_cnt += 1
                kw_flag = True
            # sklearn
            if any(sk_keyword.casefold() in code.casefold() for sk_keyword in keywords.sklearn_keywords) or \
               any(sk_keyword.casefold() in desc.casefold() for sk_keyword in keywords.sklearn_keywords):
                sklearn_cnt += 1
                kw_flag = True
            if not kw_flag:
                non_cnt += 1
        
    print('tensorflow tasks:', tf_cnt)
    print('keras tasks:', keras_cnt)
    print('pytorch tasks:', torch_cnt)
    print('sklearn tasks:', sklearn_cnt)
    print('non count:', non_cnt)
    print('total', total_cnt)
    
    return

def read_orig_juice(split):
    # dataset = [json.loads(jline) for jline in open(file_path).readlines()]
    path = 'juice-dataset/{}/orig.jsonl'.format(split)
    with open(path) as read_file:
        cnt = 0
        for jline in read_file.readlines():
            if cnt == 1: break
            # json_row = json.loads(jline)
            json_row = dict(jline)
            write_row = {}
            write_row['snippet'] = json_row['code']
            write_row['intent'] = ' '.join(json_row['nl'])
            # print(write_row['snippet'])
            # print(write_row['intent'])
            cnt += 1
            with open('juice-dataset/{}/orig.json'.format(split), 'a') as write_file:
                json.dump(write_row, write_file)
                write_file.write('\n')

    return

def total_num_of_apis():
    print('data:', len(keywords.data_process))
    print('model:', len(keywords.model_building))
    print('eval:', len(keywords.evaluation_visual))
    return

def get_data_distribution():
    data_lines = []
    
    with open('rm_dups/train/data.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))
    with open('rm_dups/test/data.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))
    with open('rm_dups/dev/data.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))

    with open('rm_dups/train/model.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))
    with open('rm_dups/test/model.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))
    with open('rm_dups/dev/model.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))

    with open('rm_dups/train/eval.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))
    with open('rm_dups/test/eval.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))
    with open('rm_dups/dev/eval.json') as datafile:
        for line in datafile.readlines():
            data_lines.append(json.loads(line))

    tfs = []
    torch = []
    keras = []
    sks = []

    for line in data_lines:
        code = line['snippet']
        if any(kw.casefold() in code.casefold() for kw in keywords.pytorch_keywords):
            torch.append(line)
        elif any(kw.casefold() in code.casefold() for kw in keywords.tensorflow_keywords):
            tfs.append(line)
        elif any(kw.casefold() in code.casefold() for kw in keywords.keras_keywords):
            keras.append(line)
        elif any(kw.casefold() in code.casefold() for kw in keywords.sklearn_keywords):
            sks.append(line)
    
    datas = 0
    models = 0
    evals = 0
    for line in tfs:
        code = line['snippet']
        if any(kw.casefold() in code.casefold() for kw in keywords.data_process):
            datas += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.model_building):
            models += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.evaluation_visual):
            evals += 1

    print('tfs:')
    print('datas:',(datas/len(tfs)))
    print('models:',(models/len(tfs)))
    print('evals:',(evals/len(tfs)))
    print(len(tfs))
    datas = 0
    models = 0
    evals = 0
    for line in torch:
        code = line['snippet']
        if any(kw.casefold() in code.casefold() for kw in keywords.data_process):
            datas += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.model_building):
            models += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.evaluation_visual):
            evals += 1

    print('\ntorch:')
    print('datas:',(datas/len(torch)))
    print('models:',(models/len(torch)))
    print('evals:',(evals/len(torch)))
    print(len(torch))
    datas = 0
    models = 0
    evals = 0
    for line in keras:
        code = line['snippet']
        if any(kw.casefold() in code.casefold() for kw in keywords.data_process):
            datas += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.model_building):
            models += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.evaluation_visual):
            evals += 1

    print('\nkeras:')
    print('datas:',(datas/len(keras)))
    print('models:',(models/len(keras)))
    print('evals:',(evals/len(keras)))
    print(len(keras))
    datas = 0
    models = 0
    evals = 0
    for line in sks:
        code = line['snippet']
        if any(kw.casefold() in code.casefold() for kw in keywords.data_process):
            datas += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.model_building):
            models += 1
        elif any(kw.casefold() in code.casefold() for kw in keywords.evaluation_visual):
            evals += 1


    print('\nsks:')
    print('datas:',(datas/len(sks)))
    print('models:',(models/len(sks)))
    print('evals:',(evals/len(sks)))
    print(len(sks))

    return

# get_loc_dist('wo_len_lim/train/data.json')
# get_loc_dist('wo_len_lim/train/feature.json')
# get_loc_dist('wo_len_lim/train/model.json')
# get_loc_dist('wo_len_lim/train/eval.json')
# get_loc_dist('wo_len_lim/train/combined.json')
# get_loc_dist('no_feature/train/data.json')


# make_dev_n_test_set('new_kw')
# make_dev_n_test_set('new_nonml', is_non_ml=True)
# get_samples_nonml('new_nonml')
# get_cnt_of_libs('new_kw')

# read_orig_juice('train')


# rm_dups_from_train('rm_dups')

# get_loc_dist('rm_dups/train/eval.json')
# total_num_of_apis()
# get_data_distribution()