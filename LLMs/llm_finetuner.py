#!/usr/bin/env python3

####################################################################################################################

# pip install transformers
# pip install peft
# pip install trl
# pip install bitsandbytes
# pip install datasets
# pip install accelerate

####################################################################################################################

import logging
import json
import shutil
import sys
import torch
import warnings
import yaml

from datasets import Dataset, DatasetDict
from huggingface_hub import login as hf_login
from pathlib import Path
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# hf_login(token=<token>)
warnings.filterwarnings('ignore')


####################################################################################################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
stream.setFormatter(formatter)
logger.addHandler(stream)

####################################################################################################################

PADDING = "-" * 150
PROMPT_PART1 = """INPUT: """
PROMPT_PART2 = """ OUTPUT: """

####################################################################################################################

bnb_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 4-bit precision base model loading
    bnb_4bit_use_double_quant=False,         # apply nested quantization for 4-bit base models (double quantization)
    bnb_4bit_quant_type='nf4',               # quantization type (fp4 or nf4)
    bnb_4bit_compute_dtype='float16',        # сompute dtype for 4-bit base models (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16')
)

####################################################################################################################

def init_tokenizer(config):
    """Загрузка и инициализация токенизатора модели."""

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.get('model_name'),
        cache_dir=config.get('hf_cache_dir', None),
    )
    tokenizer.pad_token = '<PAD>'
    tokenizer.padding_side = 'left'
    return tokenizer

####################################################################################################################

def init_model(load_config, tokenizer, config):
    """Загрузка и инициализация LLM модели."""

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.get('model_name'),
        quantization_config=load_config,
        device_map='auto',
        cache_dir=config.get('hf_cache_dir', None),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    logger.info(f'LLM FULL CONFIG:\n{PADDING}\n{model.config}\n{PADDING}\n')
    return model

####################################################################################################################

def init_peft_config(config):
    """Инициализация Lora конфига для тренировки.
    Параметры Lora: https://huggingface.co/docs/peft/package_reference/lora
    """

    peft_config = LoraConfig(
        lora_alpha=config.get('alpha', 16),
        lora_dropout=config.get('dropout', 0.1),
        r=config.get('r', 16),
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=config.get('target_modules', 'all-linear'),
    )
    return peft_config

####################################################################################################################

def init_model_train_args(work_dir, config):
    """Инициализация параметров тренировщика.
    Все параметры тренировщика: https://huggingface.co/docs/transformers/v4.45.1/en/main_classes/trainer#transformers.TrainingArguments
    Оптимизаторы: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L144
    """

    output_dir = work_dir.joinpath(config.get('common').get('output_dir'))
    all_train_args = config.get('training_args')
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=all_train_args.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=all_train_args.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=all_train_args.get('gradient_accumulation_steps', 1),
        gradient_checkpointing=all_train_args.get('gradient_checkpointing', False),
        optim=all_train_args.get('optim', 'adamw_torch'),
        save_steps=all_train_args.get('save_steps', 10),
        logging_steps=all_train_args.get('logging_steps', 10),
        learning_rate=float(all_train_args.get('learning_rate', 5e-5)),
        fp16=all_train_args.get('fp16', False),
        bf16=all_train_args.get('bf16', False),
        evaluation_strategy=all_train_args.get('evaluation_strategy', 'no'),
        eval_strategy=all_train_args.get('eval_strategy', 'no'),
        num_train_epochs=all_train_args.get('num_train_epochs', 3),
        max_steps=all_train_args.get('max_steps', -1),
        weight_decay=all_train_args.get('weight_decay', 0),
        warmup_steps=all_train_args.get('warmup_steps', 0),
        warmup_ratio=all_train_args.get('warmup_ratio', 0),
        group_by_length=all_train_args.get('group_by_length', False),
        lr_scheduler_type=all_train_args.get('lr_scheduler_type', 'linear'),
        report_to=all_train_args.get('report_to', 'none'),
        load_best_model_at_end=all_train_args.get('load_best_model_at_end', False),
        greater_is_better=all_train_args.get('greater_is_better', False),
        metric_for_best_model=all_train_args.get('metric_for_best_model', 'eval_loss'),
    )
    return training_arguments

####################################################################################################################

def init_tainer(model, dataset, peft_config, tokenizer, training_arguments, config):
    """Инициализация SFTT тренировщика.
    Все параметры SFTT тренировщика: https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer
    """

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field='text',
        max_seq_length=int(config.get('max_seq_length', 512)),
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    return trainer

####################################################################################################################

def add_instraction(source, target, gold):
    """Функция создания системного промтра для модели."""

    return ''.join((
        '<start_of_turn>user\n',
        f'{PROMPT_PART1}{source}{PROMPT_PART2}{target}<end_of_turn>\n',
        '<start_of_turn>model\n',
        f'{gold}<end_of_turn>'
    ))

####################################################################################################################

def create_dataset(work_dir, config):
    """Функция создания тренировочного и валидационного датасетов."""

    eval_data_size = config.get('eval_fraction', 0.1)
    train_src_path = work_dir.joinpath(config.get('train_source'))
    train_trg_path = work_dir.joinpath(config.get('train_target'))
    train_ref_path = work_dir.joinpath(config.get('train_reference'))

    with (
        open(train_src_path, encoding='utf8') as source,
        open(train_trg_path, encoding='utf8') as target,
        open(train_ref_path, encoding='utf8') as reference,
    ):
        raw_data = tuple(zip(map(str.strip, source), map(str.strip, target), map(str.strip, reference)))
        data = tuple(add_instraction(source=line[0], target=line[1], gold=line[2]) for line in raw_data)
        dataset = Dataset.from_dict({'text': data}).shuffle(seed=123)
        dataset = dataset.train_test_split(test_size=eval_data_size)

    return dataset

####################################################################################################################

def get_model_chat_template(model):
    messages = [ 
        { "content": "Who are you?", "role": "user" },
        { "content": "I am an LLM, bitch!", "role": "assistant" }
    ]
    template_tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_stile = template_tokenizer.apply_chat_template(messages, tokenize=False)
    logger.info(f'\nMODELS PROMPT STILE:\n{PADDING}\n{prompt_stile}\n{PADDING}')

####################################################################################################################

def main(show_model_chat_template=False):

    config_name = 'config.yml' if len(sys.argv) == 1 else sys.argv[1]
    with open(config_name) as config_yml:
        config = yaml.safe_load(config_yml)
        log_config = json.dumps(config, indent=4)

    work_dir = Path(config.get('common').get('work_dir')).resolve()

    if show_model_chat_template:
        get_model_chat_template(config.get('common').get('model_name'))
        return None

    if config.get('common').get('remove_previous'):
        llm_chkpt_dir = work_dir.joinpath(config.get('common').get('output_dir'))
        shutil.rmtree(llm_chkpt_dir, ignore_errors=True)

    logger.info(f'FINE-TUNING PARAMS:\n{PADDING}\n{log_config}\n{PADDING}\n')
    dataset = create_dataset(work_dir, config.get('data'))
    tokenizer = init_tokenizer(config.get('common'))
    model = init_model(bnb_4bit_config, tokenizer, config.get('common'))
    peft_config = init_peft_config(config.get('lora_params'))
    training_arguments = init_model_train_args(work_dir, config)
    trainer = init_tainer(model, dataset, peft_config, tokenizer, training_arguments, config.get('sftt_trainer'))
    trainer.train()
    adapter_path = work_dir.joinpath(config.get('common').get('tuned_adapter_dir'))
    trainer.model.save_pretrained(adapter_path)
    del trainer, model
    torch.cuda.empty_cache()

####################################################################################################################

if __name__ == '__main__':
    main(show_model_chat_template=False)
