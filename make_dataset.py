import csv
import json
import jsonlines
import keywords
import os
import re
import sys
from pathlib import Path

def write_on_json(out_json, nl, sc):
    temp_dict = {}
    temp_dict['snippet'] = sc
    temp_dict['intent'] = nl
    json.dump(temp_dict, out_json)
    out_json.write('\n')

def write_on_py(out_name, nl, sc):
    if os.path.exists(out_name):
        write_file = open(out_name, 'w')
    else:
        write_file = open(out_name, 'x')
    write_file.write('\'\'\'\n' + nl + '\n\'\'\'\n')
    write_file.write(sc)
    write_file.close()

def rm_comments(snippet):
    # remove py comments and % commands
        tgt_code = snippet.splitlines()
        new_code = ''
        is_comment = False
        for code_line in tgt_code:
            if '\'\'\'' in code_line or '\"\"\"' in code_line:
                if is_comment:
                    is_comment = False
                    continue
                else:
                    is_comment = True
                    continue
            if code_line.startswith('!') or code_line.startswith('%'):
                continue
            code_line = re.sub('#.*', '', code_line)
            code_line = re.sub('>>>.*', '', code_line)
            if not is_comment:
                code_line = code_line.replace('\n', '')
                if code_line and not code_line.isspace():
                    new_code += (code_line + '\n')
                # print(code_line)
    
        return new_code

def decode_ascii(data):
    is_ascii = True
    data_strip = data.replace("\n", "")

    try:
        data_strip.encode().decode('ascii')
        is_ascii = True
    except UnicodeError:
        # print('not utf-8')
        is_ascii = False
    finally:
        return is_ascii

### usage python3 make_dataset.py out_dir[/new_dir_to_save/] split[nbgrader/exercise/train]
data_split = sys.argv[2]
out_dir = './' + sys.argv[1] + '/'
header = ['file_num', 'ml_notes', 'ml_cells_in_notes',
          'ml_cells', 'total_ml_pairs',
          'total_notes', 'total_cells', 'data',
        #   'feature',
          'model', 'eval', 'combined', 'others']

Path(out_dir).mkdir(parents=True, exist_ok=True)
csvfile = open(out_dir + data_split + '_stats.csv', 'w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(header)

Path(out_dir + data_split).mkdir(parents=True, exist_ok=True)
data_json = open(out_dir + '/' + data_split + '/data.json', 'w')
# feature_json = open(out_dir + '/' + data_split + '/feature.json', 'w')
model_json = open(out_dir + '/' + data_split + '/model.json', 'w')
eval_json = open(out_dir + '/' + data_split + '/eval.json', 'w')
combined_json = open(out_dir + '/' + data_split + '/combined.json', 'w')
others_json = open(out_dir + '/' + data_split + '/others.json', 'w')


# traverse all raw notebook files in train dir
directory = './juice-notebooks/' +  data_split  + '/'
for file_counter, filename in enumerate(os.listdir(directory)):
    read_file = os.path.join(directory, filename)
    if os.path.isfile(read_file):
        # open file
        with jsonlines.open(read_file) as f:
            total_numof_cell = 0
            total_ml_pairs = 0
            ml_related_notes = 0
            ml_cells_in_notes = 0
            ml_related_cells = 0
            note_with_pair = 0
            numof_data = 0
            # numof_feature = 0
            numof_model = 0
            numof_eval = 0
            numof_combined = 0
            numof_others = 0

            # iterate each line (notebooks)
            for note_idx, note in enumerate(f.iter()):
                has_pair = False
                is_ml_related_note = False
                is_md = False
                target_description = ''
                numof_pair = 0
                if data_split == 'exercise':
                    note = json.loads(note['contents'])

                # iterate each cell 
                for cell in note['cells']:
                    cell_source = ''
                    try:
                        cell_source = str(cell['source'])
                    except(TypeError, KeyError) as e:
                        print('Error handled: '+ str(e))
                        is_md = False
                        pass
                    # check if the note is ml related
                    if any(ml_keyword.casefold() in cell_source.casefold() for ml_keyword in keywords.ml_keywords):
                        is_ml_related_note = True
                        break
                # iterate each cell and if it is ml related cell: get the NL and SC source
                for cell in note['cells']:
                    cell_type = ''
                    cell_source = ''
                    try:
                        cell_type = str(cell['cell_type'])
                        cell_source = str(cell['source'])
                    except (TypeError, KeyError) as e:
                        print('Error handled: '+ str(e))
                        is_md = False
                        pass

                    is_data = False
                    # is_feature = False
                    is_model = False
                    is_eval = False
                    
                    # get NL source
                    if (is_ml_related_note) and ('markdown' == cell_type):
                        cell_source = rm_comments(cell_source)
                        # filter 1: check for non ascii
                        if not decode_ascii(cell_source):
                            is_md = False
                            continue
                        # filter 2: length
                        if (len(cell_source) < 20) or (len(cell_source) > 150):
                            is_md = False
                            continue

                        # flagging data processing tasks
                        if any(dp_word.casefold() in cell_source.casefold() for dp_word in keywords.data_process):
                            is_data = True
                        # # flagging feature engineering tasks
                        # if any(fe_word.casefold() in cell_source.casefold() for fe_word in keywords.feature_engineering):
                        #     is_feature = True
                        # flagging model building tasks
                        if any(mb_word.casefold() in cell_source.casefold() for mb_word in keywords.model_building):
                            is_model = True
                        # flagging evaluation tasks
                        if any(ev_word.casefold() in cell_source.casefold() for ev_word in keywords.evaluation_visual):
                            is_eval = True

                        is_md = True
                        target_description = cell_source

                    # get SC source
                    if (is_ml_related_note) and ('code' == cell_type) and (is_md):
                        cell_source = rm_comments(cell_source)
                        # filter 1: check for non ascii
                        if not decode_ascii(cell_source):
                            is_md = False
                            continue
                        # filter 2: length
                        # if (len(cell_source) < 20):
                        #     is_md = False
                        #     continue

                        # filter 3: exclusion keywords
                        if any(ex_kw.casefold() in cell_source.casefold() for ex_kw in keywords.exclusion_keywords):
                            is_md = False
                            continue

                        code_lines = cell_source.splitlines()
                        numof_imports = 0
                        loc = len(code_lines)
                        is_multi_comment = False
                        for code_line in code_lines:
                            if 'import' in code_line and not is_multi_comment:
                                numof_imports += 1
                            if code_line == '':
                                loc -= 1
                        # filter 2: only imports
                        if numof_imports >= loc:
                            is_md = False
                            continue
                        if loc < 1:
                            is_md = False
                            continue
                        if loc > 10:
                            is_md = False
                            continue

                        file_num = os.path.splitext(filename)[0]
                        tasks_cnt = 0

                        example = {}
                        example['nl'] = target_description
                        example['code'] = cell_source

                        # flagging data processing tasks
                        if any(dp_word.casefold() in cell_source.casefold() for dp_word in keywords.data_process):
                            is_data = True
                            tasks_cnt += 1
                        # # flagging feature engineering tasks
                        # if any(fe_word.casefold() in cell_source.casefold() for fe_word in keywords.feature_engineering):
                        #     is_feature = True
                        #     tasks_cnt += 1
                        # flagging model building tasks
                        if any(mb_word.casefold() in cell_source.casefold() for mb_word in keywords.model_building):
                            is_model = True
                            tasks_cnt += 1
                        # flagging evaluation tasks
                        if any(ev_word.casefold() in cell_source.casefold() for ev_word in keywords.evaluation_visual):
                            is_eval = True
                            tasks_cnt += 1

                        # if task has data processing/feature engineering and contains fit/fit_transform
                        if is_data and ('.fit(' in cell_source.casefold() or ('.fit_transform(' in cell_source.casefold())):
                            tasks_cnt -= 1

                        # if more than one tasks, split as combined
                        if tasks_cnt > 1:
                            split_type = 'combined'
                            write_dir_name = out_dir + data_split + '/' + split_type + '/'
                            # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                            write_file_name = write_dir_name + file_num + '_' + \
                                              str(note_with_pair) + '_' + str(numof_pair) + '.py'
                            # write_on_py(write_file_name, target_description, cell_source)
                            write_on_json(combined_json, target_description, cell_source)
                            numof_combined += 1
                        else:
                            if is_data:
                                split_type = 'data'
                                write_dir_name = out_dir + data_split + '/' + split_type + '/'
                                # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                                write_file_name = write_dir_name + file_num + '_' + \
                                                str(note_with_pair) + '_' + str(numof_pair) + '.py'
                                # write_on_py(write_file_name, target_description, cell_source)
                                write_on_json(data_json, target_description, cell_source)
                                numof_data += 1
                            # elif is_feature:
                            #     split_type = 'feature'
                            #     write_dir_name = out_dir + data_split + '/' + split_type + '/'
                            #     # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                            #     write_file_name = write_dir_name + file_num + '_' + \
                            #                     str(note_with_pair) + '_' + str(numof_pair) + '.py'
                            #     # write_on_py(write_file_name, target_description, cell_source)               
                            #     write_on_json(feature_json, target_description, cell_source)
                            #     numof_feature += 1
                            elif is_model:
                                split_type = 'model'
                                write_dir_name = out_dir + data_split + '/' + split_type + '/'
                                # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                                write_file_name = write_dir_name + file_num + '_' + \
                                                str(note_with_pair) + '_' + str(numof_pair) + '.py'
                                # write_on_py(write_file_name, target_description, cell_source)   
                                write_on_json(model_json, target_description, cell_source)
                                numof_model += 1
                            elif is_eval:
                                split_type = 'eval'
                                write_dir_name = out_dir + data_split + '/' + split_type + '/'
                                # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                                write_file_name = write_dir_name + file_num + '_' + \
                                                str(note_with_pair) + '_' + str(numof_pair) + '.py'
                                # write_on_py(write_file_name, target_description, cell_source)
                                write_on_json(eval_json, target_description, cell_source)
                                numof_eval += 1
                            else:
                                split_type = 'others'
                                write_dir_name = out_dir + data_split + '/' + split_type + '/'
                                # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                                write_file_name = write_dir_name + file_num + '_' + \
                                                str(note_with_pair) + '_' + str(numof_pair) + '.py'
                                # write_on_py(write_file_name, target_description, cell_source)
                                write_on_json(others_json, target_description, cell_source)
                                numof_others += 1
                        is_md = False
                        has_pair = True
                        total_ml_pairs += 1
                        numof_pair += 1
                total_numof_cell += len(note['cells'])
                if has_pair:
                    note_with_pair += 1
                    has_pair = False
                if is_ml_related_note:
                    ml_related_notes += 1
                    ml_cells_in_notes += len(note['cells'])
            print(str(file_counter) + ' / ' + str(len(os.listdir(directory))))
            csvwriter.writerow([file_counter, ml_related_notes, ml_cells_in_notes, ml_related_cells,
                                total_ml_pairs, note_idx, total_numof_cell, numof_data,
                                # numof_feature,
                                numof_model, numof_eval, numof_combined, numof_others])
csvfile.close()
data_json.close()
# feature_json.close()
model_json.close()
eval_json.close()
combined_json.close()
others_json.close()
