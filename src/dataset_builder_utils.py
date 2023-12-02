# Helper functions to prepare datasets for training and evaluation.

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
import torch
from torch.utils.data import DataLoader
import numpy as np

from constants import *
from transformers import AutoTokenizer

def get_data(dataset_name, num_train_samples, num_dev_samples, num_test_samples, jointlaat = False):
    if dataset_name == 'WOS':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, 'WebOfScience'),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, 'WebOfScience'),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, 'WebOfScience')})
    elif dataset_name == 'NYT':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    # TODO change test back to 'test' from 'dev'
    elif dataset_name == 'RCV1':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    elif dataset_name == 'HCRD':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    elif dataset_name == 'CREST':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    elif dataset_name == 'MESO':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})

    label_dict = torch.load(data_path + '/' + dataset_name + '/value_dict.pt')
    num_labels = len(label_dict)
    path_list = torch.load(data_path + '/' + dataset_name + '/path_list.pt')

    if not num_train_samples == 'all':
        dataset['train'] = dataset['train'].select(range(num_train_samples))
    if not num_dev_samples == 'all':
        dataset['dev'] = dataset['dev'].select(range(num_dev_samples))
    if not num_test_samples == 'all':
        dataset['test'] = dataset['test'].select(range(num_test_samples))


    if low_resource:
        num_training_samples = len(dataset['train'])
        random_sample = np.random.choice(num_training_samples, int(num_training_samples * 0.1), replace=False)
        dataset['train'] = dataset['train'].select(random_sample)

    new_path_list = []
    for i in range(len(path_list)):
        if path_list[i][0] < num_labels:
            new_path_list.append(path_list[i])

    # Create a dict that maps first level nodes to an array of their second level nodes.
    hier_dict = {}
    for item in new_path_list:
        if item[0] not in hier_dict:
            hier_dict[item[0]] = []
        hier_dict[item[0]].append(item[1])

    if jointlaat:
        depth2label = torch.load(data_path + '/' + dataset_name + '/depth2label.pt')
        level_num_labels = []
        for level_labels in depth2label.values():
            level_num_labels.append(len(level_labels))
        return dataset, num_labels, hier_dict, level_num_labels, depth2label
    else:
        return dataset, num_labels, hier_dict

# Tokenize datasets required for training and evaluation.
def tokenize_datasets(dataset, hier_dict, num_labels):
    temp_train_datasets = tokenize_hierarchical_datasets(dataset['train'], hier_dict)
    train_datasets = []
    for i in range(len(temp_train_datasets)):
        train_datasets.append(datasets.Dataset.from_dict(temp_train_datasets[i]))
        train_datasets[i].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])

    # Validation sets split up like train sets for Early Stopping.
    temp_val_datasets = tokenize_hierarchical_datasets(dataset['dev'], hier_dict)
    val_datasets = []
    for i in range(len(temp_train_datasets)):
        val_datasets.append(datasets.Dataset.from_dict(temp_val_datasets[i]))
        val_datasets[i].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])

    # Combined validation set
    val_dataset = tokenize_flat_dataset(dataset['dev'], num_labels)
    val_dataset = datasets.Dataset.from_dict(val_dataset)
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])

    test_dataset = tokenize_flat_dataset(dataset['test'], num_labels)
    test_dataset = datasets.Dataset.from_dict(test_dataset)
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])

    return train_datasets, val_datasets, val_dataset, test_dataset

# Split up and tokenize training/validation sets for root classifier and level 1 classifiers.
def tokenize_hierarchical_datasets(dataset, hier_dict, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset_array = []
    root_data_dict = {"input_ids" : [], "attention_mask" : [], "labels" : [], "tokens" : []}
    for text, label in zip(dataset['token'], dataset['label']):
        root_data_dict['tokens'].append(text)
        tokenized_text = tokenizer(text, return_tensors = 'pt', max_length = 512, padding = 'max_length', truncation=True)
        root_data_dict["input_ids"].append(tokenized_text['input_ids'][0])
        root_data_dict["attention_mask"].append(tokenized_text['attention_mask'][0])
        root_data_dict['labels'].append(label[0])
    train_dataset_array.append(root_data_dict)

    for i in range(len(hier_dict)):
        data_dict = {"input_ids" : [], "attention_mask" : [], "labels" : [], "tokens" : []}
        for text, label in zip(dataset['token'], dataset['label']):
            if label[1] in hier_dict[i]:
                data_dict['tokens'].append(text)
                tokenized_text = tokenizer(text, return_tensors = 'pt', max_length = 512, padding = 'max_length', truncation=True)
                data_dict["input_ids"].append(tokenized_text['input_ids'][0])
                data_dict["attention_mask"].append(tokenized_text['attention_mask'][0])
                data_dict['labels'].append(hier_dict[i].index(label[1]))
        train_dataset_array.append(data_dict)
    return train_dataset_array

# Tokenize validation/test sets as a single dataset.
def tokenize_flat_dataset(dataset, num_labels, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_dict = {"input_ids" : [], "attention_mask" : [], "labels" : [], "tokens" : []}
    for text, label in zip(dataset['token'], dataset['label']):
        data_dict['tokens'].append(text)
        tokenized_text = tokenizer(text, return_tensors = 'pt', max_length = 512, padding = 'max_length', truncation=True)
        data_dict["input_ids"].append(tokenized_text['input_ids'][0])
        data_dict["attention_mask"].append(tokenized_text['attention_mask'][0])
        one_hot_labels = np.zeros(num_labels)
        one_hot_labels[label] = 1
        data_dict['labels'].append(torch.from_numpy(np.float32(one_hot_labels)))
    return data_dict

def convert_to_dataloaders(train_datasets, val_datasets, val_dataset, test_dataset, train_batch_size, test_batch_size):
    train_loaders = []
    for i in range(len(train_datasets)):
        train_loaders.append(DataLoader(train_datasets[i], batch_size = train_batch_size))

    val_loaders = []
    for i in range(len(val_datasets)):
        val_loaders.append(DataLoader(val_datasets[i], batch_size = test_batch_size))
        
    val_loader = DataLoader(val_dataset, batch_size = test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size)
    return train_loaders, val_loaders, val_loader, test_loader

# Helper methods for flat classifier dataset building.
def tokenize_datasets_flat(dataset, num_labels, model_name):
    train_dataset = tokenize_flat_dataset(dataset['train'], num_labels, model_name)
    train_dataset = datasets.Dataset.from_dict(train_dataset)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])

    val_dataset = tokenize_flat_dataset(dataset['dev'], num_labels, model_name)
    val_dataset = datasets.Dataset.from_dict(val_dataset)
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])

    test_dataset = tokenize_flat_dataset(dataset['test'], num_labels, model_name)
    test_dataset = datasets.Dataset.from_dict(test_dataset)
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'tokens'])
    return train_dataset, val_dataset, test_dataset

def convert_to_dataloaders_flat(train_dataset, val_dataset, test_dataset, train_batch_size, test_batch_size):
    train_loader = DataLoader(train_dataset, batch_size = train_batch_size)
    val_loader = DataLoader(val_dataset, batch_size = test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size)
    return train_loader, val_loader, test_loader