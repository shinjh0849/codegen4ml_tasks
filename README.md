# Data, code, model used for the paper "The Good, the Bad, and the Missing: Neural Code Generation for Machine Learning Tasks"


## 1. Dataset
The raw dataset we constructed for different ML tasks and the non-ml tasks in json files are in raw_dataset folder


split_of_training = ['train', 'dev', 'test']

split_of_tasks = ['data', 'model', 'eval', 'noml', 'uni']

raw_dataset/[split_of_training]_[split_of_tasks].json



## 2. Using the 6 baseline models

### 0. We provide the snap shot of the models that we have trained.
This is the link to download the model snap shot: [link](https://drive.google.com/drive/folders/1M87JktTwvZ64lkWPkQ4Och0zeqYrJDR7?usp=sharing).



### 1. tranX

#### 1. clone the repo

    git clone https://github.com/pcyin/tranX.git
    cd tranX

#### 2. binarize the json data

1. save the json files to a directory in the repository: e.g. datasets/conala/data/conala/

2. using datasets/conala/dataset.py, binarize the json file.

    ```python dataset.py ```   

#### 3. load the snap shot of the models.

1. save the trained model file to a directory in the repository: e.g. saved_models/

    model files: tranx_train[split_of_tasks].bin

2. run the testing script to get the decoded results (match the path to your corresponding locations).

    ```bash scripts/conala/decode.sh [path_of_binarizezd data] [path_of_saved models]  ```

3. the decoded file will be saved in "decodes/conala/\$(basename \${model_file}).\$(basename \${decode_file}).decode".

#### 4. For training and more details, refer to the original repository.

<br>

### 2. EK Codegen

#### 1. clone the repo

    git clone https://github.com/neulab/external-knowledge-codegen.git
    cd external-knowledge-codegen

#### 2. binarize the json data

1. save the json files to a directory in the repository: e.g. datasets/conala/

2. use the following command to binarize the json file.

    ```python dataset.py --pretrain data/conala/conala-mined.jsonl --topk 100000 --include_api apidocs/processed/distsmpl/snippet_15k/goldmine_snippet_count100k_topk1_temp2.jsonl```

#### 3. load the snap shot of the models.

1. save the trained model file to a diretory in the repository e.g. saved_models/

    pretrained model files: ek_juice_pretrain_[split_of_tasks].bin
    finetuned model files: ek_finetune.[splits_of_tasks].bin

2. run the testing script to get the decoded results.

    ```bash scripts/conala/test.sh [path_of_saved models]```

3. the decoded fill will be saved in "decodes/conala/$(basename $1).test.decode".

#### 4. For training and more details, refer to the original repository.

<br>

### 3. CG-RL

#### 1. clone the repo

    git clone https://github.com/DeepLearnXMU/CG-RL.git
    cd CG-RL

#### 2. binarize the json data

1. save the json files to a directory in the repository: e.g. datasets/conala/

2. using datasets/conala/dataset.py in the tranX reposotory, binarize the json file.

    ```python dataset.py ```   

#### 3. load the snap shot of the models.

1. save the trained model file to a diretory in the repository e.g. saved_models/

    pretrained model files: CG_RL.juice.pretrain.[split_of_tasks].bin
    finetuned model files: CG_RL.train_rl.[splits_of_tasks].bin

2. run the testing script to get the decoded results.

    ```bash scripts/conala/test.sh ```

3. the decoded fill will be saved in "decodes/conala/$(basename $1).test.decode".

#### 4. For training and more details, refer to the original repository.

<br>

### 4. TreeCodeGen

#### 1. clone the repo

    git clone https://github.com/sdpmas/TreeCodeGen.git
    cd TreeCodeGen

#### 2. binarize the json data

1. save the json files to a directory in the repository: 

2. Build a file with NL intents of training set: Refer to datasets/conala/retrive_src.py. The file will be saved as src.txt in data/conala

3. Parse those NL intents and build a vocabulary: refer to https://github.com/nxphi47/tree_transformer for more details on setting up the parser and run convert_ln.py

4. Build train/dev/test dataset: Run datasets/conala/dataset_hie.py


#### 3. load the snap shot of the models.

1. save the trained model file to a diretory in the repository e.g. saved_models/

    model files: tree_best.[split_of_tasks]_.bin

2. run the testing script to get the decoded results.

    ```bash scripts/conala/test.sh ```

#### 4. For training and more details, refer to the original repository.

<br>

### 5. Code-gen-TAE

#### 1. clone the repo

    git clone https://github.com/BorealisAI/code-gen-TAE.git
    cd code-gen-TAE
    pip install -r requirements.txt

#### 2. binarize the json data

1. save the json files to a directory in the repository: e.g. datasets/conala/

2. using datasets/conala/dataset.py in the tranX reposotory, binarize the json file.

    ```python dataset.py ```   

#### 3. load the snap shot of the models.

1. save the trained model file to a diretory in the repository e.g. saved_models/

    model files: tae_[split_of_tasks]_resume.pth

2. run the testing script to get the decoded results.

    ```python3 train.py --dataset_name conala --save_dir saved_models/ --copy_bt --no_encoder_update --monolingual_ratio 0.5 --epochs 20 --just_evaluate --seed 4```

#### 4. For training and more details, refer to the original repository.

<br>

### 6. PyCodeGPT

#### 1. clone the repo

    git clone https://github.com/microsoft/PyCodeGPT.git
    cd PyCodeGPT
    pip install -r requirements.txt
    cd cert
    pip install -r requirements.txt

#### 2. binarize the json data

1. save the json files to a directory in the repository

2. binarize the json file.

    ```bash run_encode_domain.sh ```   

#### 3. load the snap shot of the models.

1. save the trained model file to a diretory in the repository e.g. saved_models/

    model files (12 files): pcgpt_[split_of_tasks]_*

2. run the testing script to get the decoded results.

    ```cd ../```

    ```
    !python eval_human_eval.py \
	--model_name_or_path [path_to_model_files] \
	--output_dir [output_dir] \
	--num_completions [number_of_generations] \
	--temperature 0.6 \
	--top_p 0.95 \
	--max_new_tokens 100 \
	--gpu_device 0 \
    ```

#### 4. For training and more details, refer to the original repository.

<br>

## Additional scripts used for the study.

```bash
├── keywords.py (keywords used to parse ML APIs from the libraries)
├── make_dataset.py (script to construct the different ML task dataset)
├── make_non_m.py (script to construct the non-ml dataset)
├── utils.py (script to clean and split the dataset)
```