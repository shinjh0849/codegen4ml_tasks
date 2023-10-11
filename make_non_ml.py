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
header = ['file_num', 'total_notes', 'total_cells', 'non_ml']

Path(out_dir).mkdir(parents=True, exist_ok=True)
csvfile = open(out_dir + data_split + '_stats.csv', 'w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(header)

Path(out_dir + data_split).mkdir(parents=True, exist_ok=True)
non_ml_json = open(out_dir + '/' + data_split + '/non_ml.json', 'w')



# traverse all raw notebook files in train dir
directory = './juice-notebooks/' +  data_split  + '/'
for file_counter, filename in enumerate(os.listdir(directory)):
    read_file = os.path.join(directory, filename)
    if os.path.isfile(read_file):
        # open file
        with jsonlines.open(read_file) as f:
            total_numof_cell = 0
            note_with_pair = 0
            numof_non_ml = 0

            # iterate each line (notebooks)
            for note_idx, note in enumerate(f.iter()):
                is_ml_related_note = False
                has_pair = False
                is_md = False
                target_description = ''
                numof_pair = 0
                if data_split == 'exercise':
                    note = json.loads(note['contents'])

                # iterate each cell and if it is ml related cell: if is ml related note, flag
                for cell in note['cells']:
                    cell_type = ''
                    cell_source = ''
                    try:
                        cell_source = str(cell['source'])
                    except (TypeError, KeyError) as e:
                        print('Error handled: '+ str(e))
                        is_md = False
                        pass
                        
                    # check if the NL cell is ml related
                    if any(ml_keyword.casefold() in cell_source.casefold() for ml_keyword in keywords.ml_keywords) or \
                        any(dp_word.casefold() in cell_source.casefold() for dp_word in keywords.data_process) or \
                        any(mb_word.casefold() in cell_source.casefold() for mb_word in keywords.model_building) or \
                        any(ev_word.casefold() in cell_source.casefold() for ev_word in keywords.evaluation_visual) or \
                        any(ek_word.casefold() in cell_source.casefold() for ek_word in keywords.exclusion_keywords):
                        is_ml_related_note = True
                        break

                # iterate each cell and if it is ml related cell: get the NL and SC source
                for cell in note['cells']:
                    is_ml_related_cell = True
                    cell_type = ''
                    cell_source = ''
                    try:
                        cell_type = str(cell['cell_type'])
                        cell_source = str(cell['source'])
                    except (TypeError, KeyError) as e:
                        print('Error handled: '+ str(e))
                        is_md = False
                        pass

                    # check if the NL cell is ml related
                    if any(ml_keyword.casefold() in cell_source.casefold() for ml_keyword in keywords.ml_keywords) or \
                        any(dp_word.casefold() in cell_source.casefold() for dp_word in keywords.data_process) or \
                        any(mb_word.casefold() in cell_source.casefold() for mb_word in keywords.model_building) or \
                        any(ev_word.casefold() in cell_source.casefold() for ev_word in keywords.evaluation_visual) or \
                        any(ek_word.casefold() in cell_source.casefold() for ek_word in keywords.exclusion_keywords):
                        is_ml_related_cell = True
                    else:
                        is_ml_related_cell = False
                    
                    # get NL source
                    if (not is_ml_related_note) and (not is_ml_related_cell) and ('markdown' == cell_type):
                        cell_source = rm_comments(cell_source)
                        # filter 1: check for non ascii
                        if not decode_ascii(cell_source):
                            is_md = False
                            continue
                        # filter 2: length
                        if (len(cell_source) < 20) or (len(cell_source) > 150):
                            is_md = False
                            continue

                        is_md = True
                        target_description = cell_source

                    # get SC source
                    if (not is_ml_related_note) and (not is_ml_related_cell) and ('code' == cell_type) and (is_md):
                        cell_source = rm_comments(cell_source)
                        # filter 1: check for non ascii
                        if not decode_ascii(cell_source):
                            is_md = False
                            continue

                        if len(cell_source) < 20:
                            is_md = False
                            continue

                        # check if the SC cell is ml related
                        if any(ml_keyword.casefold() in cell_source.casefold() for ml_keyword in keywords.ml_keywords) or \
                            any(dp_word.casefold() in cell_source.casefold() for dp_word in keywords.data_process) or \
                            any(mb_word.casefold() in cell_source.casefold() for mb_word in keywords.model_building) or \
                            any(ev_word.casefold() in cell_source.casefold() for ev_word in keywords.evaluation_visual) or \
                            any(ek_word.casefold() in cell_source.casefold() for ek_word in keywords.exclusion_keywords):
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


                        # if more than one tasks, split as combined
                        split_type = 'non_ml'
                        write_dir_name = out_dir + data_split + '/' + split_type + '/'
                        # Path(write_dir_name).mkdir(parents=True, exist_ok=True)
                        write_file_name = write_dir_name + file_num + '_' + \
                                            str(note_with_pair) + '_' + str(numof_pair) + '.py'
                        # write_on_py(write_file_name, target_description, cell_source)
                        write_on_json(non_ml_json, target_description, cell_source)
                        numof_non_ml += 1
                        
                        is_md = False
                        has_pair = True
                        numof_pair += 1
                total_numof_cell += len(note['cells'])
                if has_pair:
                    note_with_pair += 1
                    has_pair = False
 
            print(str(file_counter) + ' / ' + str(len(os.listdir(directory))))
            header = ['file_num', 'total_notes', 'total_cells', 'non_ml']

            csvwriter.writerow([file_counter, note_idx, total_numof_cell, numof_non_ml])
csvfile.close()
non_ml_json.close()
