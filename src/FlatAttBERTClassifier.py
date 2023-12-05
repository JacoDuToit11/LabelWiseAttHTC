import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.classification import MultilabelStatScores
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
torch.set_float32_matmul_precision('medium')

import wandb
from pytorch_lightning.loggers import WandbLogger

from AttBERTmodels import *
from dataset_builder_utils import *

dataset_names = ['WOS', 'NYT', 'RCV1']

# test = True
test = False

# use_wandb = test
use_wandb = not test

hyperparam_tune = False
# hyperparam_tune = True

# load_model_from_file = True
load_model_from_file = False

layer_wise_lr_decay = True
# layer_wise_lr_decay = False

linear_lr_decay = True
# linear_lr_decay = False

linear_warmup = True
# linear_warmup = False

linear_warmup_steps_percentage = 0.1
lr_decay_min_ratio = 0.1

laat_projection_dim = 256

use_lr_scheduler = False
# use_lr_scheduler = True

use_different_lrs = False
# use_different_lrs = True

# stopping_criteria = 'macro-f1'
stopping_criteria = 'harmonic_mean_micro_macro_f1'

if use_different_lrs:
    bert_lrs = [1e-5, 2e-5]
    attention_lrs = [1e-4, 5e-4, 1e-3]

if test:
    num_epochs = 1
    learning_rates = [1e-4]
    layer_wise_decay_ratio = 0.8
    num_epochs_for_decaying = 10
    batch_sizes = [5]
    random_seeds = [33]
    num_train_samples = 50
    num_dev_samples = 20
    num_test_samples = 20
else:
    if hyperparam_tune:
        if use_lr_scheduler:
            num_epochs = 10
            learning_rates = [1e-4, 7.5e-5, 5e-5, 2.5e-5]
        else:
            num_epochs = 20
            learning_rates = [2e-5]

        if layer_wise_lr_decay:
            if low_resource:
                num_epochs = 22
                num_epochs_for_decaying = 20
            else:
                num_epochs = 12
                num_epochs_for_decaying = 10
            
            learning_rates = [3e-4, 2e-4, 1e-4, 7.5e-5]
            layer_wise_decay_ratio = 0.8
        batch_sizes = [16]
        random_seeds = [44]
    else:
        # Final runs!
        if use_lr_scheduler:
            num_epochs = 10
            random_seeds = [33, 44, 55]
        if layer_wise_lr_decay:
            if low_resource:
                num_epochs = 22
                num_epochs_for_decaying = 20
            else:
                num_epochs = 12
                num_epochs_for_decaying = 10
            layer_wise_decay_ratio = 0.8
            random_seeds = [44]


    num_train_samples = 'all'
    num_dev_samples = 'all'
    num_test_samples = 'all'

cnn_model_types = ['FlatBERTAttModel', 'FlatBERTLaatModel']

linear_warmup_ratio = 0.25

test_batch_size = 32
earlystop_patience = 5
dropout_probs = [0.0]

if test:
    model_name = 'prajjwal1/bert-tiny'
    # model_name = 'prajjwal1/bert-medium'
else:
    model_name = 'bert-base-uncased'
    # model_name = 'roberta-base'

# concat_last_4_hidden_states = True
concat_last_4_hidden_states = False

def main():
    # global laat_projection_dim
    global cnn_model_type
  
    for dataset_name in dataset_names:
        dataset, num_labels, hier_dict, level_num_labels, depth2label = get_data(dataset_name, num_train_samples, num_dev_samples, num_test_samples, True)
        train_dataset, val_dataset, test_dataset = tokenize_datasets_flat(dataset, num_labels, model_name)
        for temp_cnn_model_type in cnn_model_types:
            cnn_model_type = temp_cnn_model_type
            for index, random_seed in enumerate(random_seeds):
                pl.seed_everything(random_seed, workers=True)
                if hyperparam_tune:
                    for dropout_prob_index in range(len(dropout_probs)):
                        dropout_prob = dropout_probs[dropout_prob_index]
                        for batch_size in batch_sizes:
                            if use_different_lrs:
                                for bert_lr in bert_lrs:
                                    for attention_lr in attention_lrs:
                                        train(dataset_name, train_dataset, val_dataset, test_dataset, num_labels, depth2label, random_seed, batch_size, 0, dropout_prob, bert_lr, attention_lr)
                            else:
                                for learning_rate in learning_rates:
                                    train(dataset_name, train_dataset, val_dataset, test_dataset, num_labels, depth2label, random_seed, batch_size, learning_rate, dropout_prob)
                else:
                    if low_resource:
                        if dataset_name == 'WOS':
                            if cnn_model_type == 'FlatBERTAttModel':
                                batch_size = 16
                                learning_rate = 1e-4
                            elif cnn_model_type == 'FlatBERTLaatModel':
                                batch_size = 16
                                learning_rate = 1e-4
                        elif dataset_name == 'NYT':
                            if cnn_model_type == 'FlatBERTAttModel':
                                batch_size = 16
                                learning_rate = 2e-4
                            elif cnn_model_type == 'FlatBERTLaatModel':
                                batch_size = 16
                                learning_rate = 2e-4
                        elif dataset_name == 'RCV1':
                            if cnn_model_type == 'FlatBERTAttModel':
                                batch_size = 16
                                learning_rate = 2e-4
                            elif cnn_model_type == 'FlatBERTLaatModel':
                                batch_size = 16
                                learning_rate = 1e-4
                    else:
                        # Final runs
                        if dataset_name == 'WOS':
                            if cnn_model_type == 'FlatBERTAttModel':
                                batch_size = 16
                                learning_rate = 7.5e-5
                                laat_projection_dim = 0
                            elif cnn_model_type == 'FlatBERTLaatModel':
                                batch_size = 32
                                learning_rate = 2e-4
                                laat_projection_dim = 512
                        elif dataset_name == 'NYT':
                            if cnn_model_type == 'FlatBERTAttModel':
                                batch_size = 16
                                learning_rate = 2e-4
                                laat_projection_dim = 0
                            elif cnn_model_type == 'FlatBERTLaatModel':
                                batch_size = 16
                                learning_rate = 1e-4
                                laat_projection_dim = 512
                        elif dataset_name == 'RCV1':
                            if cnn_model_type == 'FlatBERTAttModel':
                                batch_size = 16
                                learning_rate = 7.5e-5
                                laat_projection_dim = 0
                            elif cnn_model_type == 'FlatBERTLaatModel':
                                batch_size = 16
                                learning_rate = 2e-4
                                laat_projection_dim = 512
                    train(dataset_name, train_dataset, val_dataset, test_dataset, num_labels, depth2label, random_seed, batch_size, learning_rate, 0)

def train(dataset_name, train_dataset, val_dataset, test_dataset, num_labels, depth2label, random_seed, train_batch_size, learning_rate, dropout_prob, bert_lr = 0, attention_lr = 0):
    # Build datasets.
    train_loader, val_loader, test_loader = convert_to_dataloaders_flat(train_dataset, val_dataset, test_dataset, train_batch_size, test_batch_size)
    global num_batches_per_epoch
    num_batches_per_epoch = len(train_loader)

    if cnn_model_type == 'FlatBERTLaatModel':
        classifier = FlatBERTLaatModel(num_labels, model_name, concat_last_4_hidden_states, laat_projection_dim, dropout_prob)
    elif cnn_model_type == 'FlatBERTAttModel':
        classifier = FlatBERTAttModel(num_labels, model_name, concat_last_4_hidden_states, dropout_prob)
    
    global training
    training = True
    lightning_model = LightningModel(model=classifier, learning_rate=learning_rate, num_labels=num_labels, depth2label = depth2label, bert_lr = bert_lr, attention_lr = attention_lr)
    
    if stopping_criteria == 'harmonic_mean_micro_macro_f1':
        early_stop_callback = EarlyStopping(monitor="val_harmonic_mean_micro_macro_f1", mode="max", min_delta=0.00, patience=earlystop_patience)
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename = "model_flat", monitor="val_harmonic_mean_micro_macro_f1", mode="max", save_weights_only=True)
    elif stopping_criteria == 'macro_f1':
        early_stop_callback = EarlyStopping(monitor="val_macro_f1", mode="max", min_delta=0.00, patience=earlystop_patience)
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename = "model_flat", monitor="val_macro_f1", mode="max", save_weights_only=True)
    if use_wandb:
        wandb_logger = WandbLogger(project = "Research Classification", name = "Flat")
        wandb_logger.experiment.config.update({'dataset': dataset_name, 'classifier': 'hier_local', 'bert_model': model_name, 'batch_size': train_batch_size, 'learning_rate': learning_rate, 'dropout_prob': dropout_prob,
                                                'random_seed': random_seed, 'earlystop_patience': earlystop_patience, 
                                                'concat_last_4_hidden_states': concat_last_4_hidden_states, 'cnn_model_type': cnn_model_type, 'laat_projection_dim': laat_projection_dim, 
                                                'use_lr_scheduler': use_lr_scheduler, 'bert_lr': bert_lr, 'attention_lr': attention_lr, 'linear_warmup_ratio': linear_warmup_ratio,
                                                'linear_lr_decay': linear_lr_decay, 'layer_wise_lr_decay': layer_wise_lr_decay, 'layer_wise_decay_ratio': layer_wise_decay_ratio,
                                               'linear_warmup': linear_warmup, 'linear_warmup_steps_percentage': linear_warmup_steps_percentage, 'lr_decay_min_ratio': lr_decay_min_ratio, 
                                               'stopping_criteria': stopping_criteria, 'low_resource': low_resource})
        trainer = pl.Trainer(precision="16-mixed", max_epochs=num_epochs, deterministic=True, accelerator=accelerator, callbacks=[early_stop_callback, 
                        checkpoint_callback], logger = wandb_logger)
    else:
        if use_lr_scheduler:
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            trainer = pl.Trainer(precision="16-mixed", max_epochs=num_epochs, deterministic=True, accelerator=accelerator, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor])
        else:
            trainer = pl.Trainer(precision="16-mixed", max_epochs=num_epochs, deterministic=True, accelerator=accelerator, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Load best model from checkpoint.
    checkpoint_path = checkpoint_callback.best_model_path
    if use_wandb:
        wandb_logger.experiment.config.update({'checkpoint_path': checkpoint_path})
    checkpoint_weights = torch.load(checkpoint_path)['state_dict']
    keys = list(checkpoint_weights.keys())
    for key in keys:
        checkpoint_weights[key[6:]] = checkpoint_weights.pop(key)
    classifier.load_state_dict(checkpoint_weights)

    lightning_model = LightningModel(model=classifier, learning_rate=learning_rate, num_labels=num_labels, depth2label = depth2label, bert_lr = bert_lr, attention_lr = attention_lr)
    training = False
    trainer.validate(model = lightning_model, dataloaders = val_loader)
    # Load best model from checkpoint.
    training = False
    trainer.validate(model = lightning_model, dataloaders = val_loader)
    trainer.test(model = lightning_model, dataloaders = test_loader)
    wandb.finish()

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate, num_labels, depth2label, bert_lr, attention_lr):
        super().__init__()
        self.model = model
        
        self.learning_rate = learning_rate
        self.bert_lr = bert_lr
        self.attention_lr = attention_lr
        self.num_labels = num_labels
        self.depth2label = depth2label

        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.val_epoch_outputs = []
        self.val_epoch_labels = []
        self.test_epoch_outputs = []
        self.test_epoch_labels = []

    def obtain_output(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        label_logits = self.obtain_output(batch)
        loss = self.loss_fn(label_logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.obtain_output(batch)
        labels = batch['labels']
        
        self.val_epoch_outputs.append(output)
        self.val_epoch_labels.append(labels)
    
    def on_validation_epoch_end(self):
        val_epoch_outputs = torch.cat(self.val_epoch_outputs, dim = 0)
        val_epoch_labels = torch.cat(self.val_epoch_labels, dim = 0)

        loss = self.loss_fn(val_epoch_outputs, val_epoch_labels)
        self.log("val_loss", loss)
        val_epoch_outputs = torch.sigmoid(val_epoch_outputs)
        self.log_metrics(val_epoch_outputs, val_epoch_labels, 'val_')

        self.val_epoch_outputs.clear()
        self.val_epoch_labels.clear()
    
    def test_step(self, batch, batch_idx):
        output = self.obtain_output(batch)
        labels = batch['labels']

        self.test_epoch_outputs.append(output)
        self.test_epoch_labels.append(labels)
    
    def on_test_epoch_end(self):
        test_epoch_outputs = torch.cat(self.test_epoch_outputs, dim = 0)
        test_epoch_labels = torch.cat(self.test_epoch_labels, dim = 0)

        test_epoch_outputs = torch.sigmoid(test_epoch_outputs)
        self.log_metrics(test_epoch_outputs, test_epoch_labels, 'test_')

        self.test_epoch_outputs.clear()
        self.test_epoch_labels.clear()
    
    def log_metrics(self, output, labels, phase):
        total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class = self.get_stats(output, labels)

        micro_f1 = self.log_micro_metrics(total_TP, total_FP, total_FN, phase)
        macro_f1 = self.log_macro_metrics(TP_per_class, FP_per_class, FN_per_class, phase)

        self.log_harmonic_mean_micro_macro_f1(micro_f1, macro_f1, phase)

        if phase == 'test_':
            self.log_scores_per_level(TP_per_class, FP_per_class, FN_per_class, phase)

    def log_scores_per_level(self, TP_per_class, FP_per_class, FN_per_class, phase):
        TP_per_class = np.array(TP_per_class)
        FP_per_class = np.array(FP_per_class)
        FN_per_class = np.array(FN_per_class)

        for i in range(len(self.depth2label)):
            TP_per_class_level_i = TP_per_class[self.depth2label[i]]
            FP_per_class_level_i = FP_per_class[self.depth2label[i]]
            FN_per_class_level_i = FN_per_class[self.depth2label[i]]

            # Micro-F1 scores for this level
            total_TP_level_i = sum(TP_per_class_level_i)
            total_FP_level_i = sum(FP_per_class_level_i)
            total_FN_level_i = sum(FN_per_class_level_i)

            if total_TP_level_i + total_FP_level_i > 0:
                micro_precision =  total_TP_level_i/(total_TP_level_i + total_FP_level_i)
            else:
                micro_precision = 0
            if total_TP_level_i + total_FN_level_i > 0:
                micro_recall =  total_TP_level_i/(total_TP_level_i + total_FN_level_i)
            else:
                micro_recall = 0
            if micro_precision + micro_recall > 0:
                micro_f1 = 2*((micro_precision * micro_recall)/(micro_precision + micro_recall))
            else:
                micro_f1 = 0

            self.log(phase + 'level' + str(i + 1) + '_micro_prec', micro_precision, on_step=False, on_epoch=True)
            self.log(phase + 'level' + str(i + 1) + '_micro_recall', micro_recall, on_step=False, on_epoch=True) 
            self.log(phase + 'level' + str(i + 1) + '_micro_f1', micro_f1, on_step=False, on_epoch=True)

            # Macro-F1 scores for this level
            precision_per_class_level_i = []
            recall_per_class_level_i = []
            f1_per_class_level_i = []
            for class_index in range(len(TP_per_class_level_i)):
                if TP_per_class_level_i[class_index] + FP_per_class_level_i[class_index] != 0:
                    temp_precision = TP_per_class_level_i[class_index]/(TP_per_class_level_i[class_index] + FP_per_class_level_i[class_index]) 
                else:
                    temp_precision = 0
                precision_per_class_level_i.append(temp_precision)

                if TP_per_class_level_i[class_index] + FN_per_class_level_i[class_index] != 0:
                    temp_recall = TP_per_class_level_i[class_index]/(TP_per_class_level_i[class_index] + FN_per_class_level_i[class_index])
                else:
                    temp_recall = 0
                recall_per_class_level_i.append(temp_recall)

                if temp_precision + temp_recall > 0:
                    temp_f1 = 2*((temp_precision * temp_recall)/(temp_precision + temp_recall))
                else:
                    temp_f1 = 0
                f1_per_class_level_i.append(temp_f1)

            macro_precision = torch.mean(torch.Tensor(precision_per_class_level_i)).item()
            macro_recall = torch.mean(torch.Tensor(recall_per_class_level_i)).item()
            macro_f1 = torch.mean(torch.Tensor(f1_per_class_level_i)).item()

            self.log(phase + 'level' + str(i + 1) + '_macro_prec', macro_precision, on_step=False, on_epoch=True)
            self.log(phase + 'level' + str(i + 1) + '_macro_recall', macro_recall, on_step=False, on_epoch=True) 
            self.log(phase + 'level' + str(i + 1) + '_macro_f1', macro_f1, on_step=False, on_epoch=True)
    
    def log_harmonic_mean_micro_macro_f1(self, micro_f1, macro_f1, phase):
        if micro_f1 + macro_f1 > 0:
            harmonic_mean = (2 * micro_f1 * macro_f1)/(micro_f1 + macro_f1)
        else: 
            harmonic_mean = 0
        self.log(phase + 'harmonic_mean_micro_macro_f1', harmonic_mean, on_step=False, on_epoch=True)

    def get_stats(self, output, labels):
        threshold = 0.5
        
        output = output.to(self.device)
        labels = labels.to(self.device)
        stat_scores_metric = MultilabelStatScores(average=None, num_labels = self.num_labels, threshold = threshold).to(self.device)
            
        stat_scores = stat_scores_metric(output, labels)
        TP_per_class = stat_scores[:, 0].tolist()
        FP_per_class = stat_scores[:, 1].tolist()
        FN_per_class = stat_scores[:, 3].tolist()

        total_TP = sum(TP_per_class)
        total_FP = sum(FP_per_class)
        total_FN = sum(FN_per_class)

        return total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class

    def log_micro_metrics(self, total_TP, total_FP, total_FN, phase):
        if total_TP + total_FP > 0:
            micro_precision =  total_TP/(total_TP + total_FP)
        else:
            micro_precision = 0
        if total_TP + total_FN > 0:
            micro_recall =  total_TP/(total_TP + total_FN)
        else:
            micro_recall = 0
        if micro_precision + micro_recall > 0:
            micro_f1 = 2*((micro_precision * micro_recall)/(micro_precision + micro_recall))
        else:
            micro_f1 = 0

        self.log(phase + 'micro_prec', micro_precision, on_step=False, on_epoch=True)
        self.log(phase + 'micro_recall', micro_recall, on_step=False, on_epoch=True) 
        self.log(phase + 'micro_f1', micro_f1, on_step=False, on_epoch=True)

        return micro_f1

    def log_macro_metrics(self, TP_per_class, FP_per_class, FN_per_class, phase):
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for i in range(len(TP_per_class)):
            if TP_per_class[i] + FP_per_class[i] != 0:
                temp_precision = TP_per_class[i]/(TP_per_class[i] + FP_per_class[i]) 
            else:
                temp_precision = 0
            precision_per_class.append(temp_precision)

            if TP_per_class[i] + FN_per_class[i] != 0:
                temp_recall = TP_per_class[i]/(TP_per_class[i] + FN_per_class[i])
            else:
                temp_recall = 0
            recall_per_class.append(temp_recall)

            if temp_precision + temp_recall > 0:
                temp_f1 = 2*((temp_precision * temp_recall)/(temp_precision + temp_recall))
            else:
                temp_f1 = 0
            f1_per_class.append(temp_f1)
        
        macro_precision = torch.mean(torch.Tensor(precision_per_class)).item()
        macro_recall = torch.mean(torch.Tensor(recall_per_class)).item()
        macro_f1 = torch.mean(torch.Tensor(f1_per_class)).item()

        self.log(phase + 'macro_prec', macro_precision, on_step=False, on_epoch=True)
        self.log(phase + 'macro_recall', macro_recall, on_step=False, on_epoch=True) 
        self.log(phase + 'macro_f1', macro_f1, on_step=False, on_epoch=True)

        class_ids = [(i+1) for i in range(len(f1_per_class))]
        macro_f1_table_data = [[class_id, f1_score] for (class_id, f1_score) in zip(class_ids, f1_per_class)]
        macro_f1_table = wandb.Table(data = macro_f1_table_data, columns=["class", "F1-score"])
        wandb.log({phase + ' F1-score per class': wandb.plot.bar(macro_f1_table, "class", "F1-score", title = phase + " F1-score per class")})

        return macro_f1

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if layer_wise_lr_decay:
            if epoch == 0 and batch_idx == 0:
                    self.orig_learning_rates = []
                    for pg in optimizer.param_groups:
                        self.orig_learning_rates.append(pg['lr'])
            if epoch == 0:
                if linear_warmup:
                    self.last_warmup_step = int(num_batches_per_epoch * linear_warmup_steps_percentage)
                    if batch_idx < self.last_warmup_step:
                        lr_scale = min(1.0, (float(batch_idx + 1)/self.last_warmup_step))
                        for idx, pg in enumerate(optimizer.param_groups):
                            pg['lr'] = self.orig_learning_rates[idx] * lr_scale
            else:
                if linear_lr_decay:
                    if batch_idx == 0:
                        for idx, pg in enumerate(optimizer.param_groups):
                            pg['lr'] = self.orig_learning_rates[idx] * max((1 - (epoch/num_epochs_for_decaying)), lr_decay_min_ratio)
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def configure_optimizers(self):
        if layer_wise_lr_decay:
            layer_names = []
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                layer_names.append(name)
            layer_names.reverse()

            parameters = []
            lr = self.learning_rate
            prev_group_name = 'attention_layer'
            for idx, name in enumerate(layer_names):
                if cnn_model_type == 'FlatBERTAttModel':
                    if name.split('.')[0] in 'WQ':
                        curr_group_name = 'attention_layer'
                elif cnn_model_type == 'FlatBERTLaatModel':
                    if name.split('.')[0] in 'WU' or name.split('.')[0] in 'final_linear':
                        curr_group_name = 'attention_layer'
                if 'bert.encoder.layer' in name:
                    curr_group_name = name[0:20]
                elif 'bert.embeddings' in name:
                    curr_group_name = 'embeddings'

                if curr_group_name != prev_group_name:
                    lr *= layer_wise_decay_ratio
                prev_group_name = curr_group_name
    
                # append layer parameters
                parameters += [{'params': [p for n, p in self.model.named_parameters() if n == name and p.requires_grad], 'lr': lr}]
            optimizer = Adam(parameters)
            return optimizer
        else:
            if use_different_lrs:
                if cnn_model_type == 'FlatBERTLaatModel':
                    optimizer = Adam([{'params': self.model.bert.parameters(), 'lr':self.bert_lr}, 
                                    {'params': self.model.W.parameters(), 'lr':self.attention_lr}, 
                                    {'params': self.model.U.parameters(), 'lr':self.attention_lr}, 
                                    {'params': self.model.final_linear.parameters(), 'lr':self.attention_lr}])
                elif cnn_model_type == 'FlatBERTAttModel':
                    optimizer = Adam([{'params': self.model.bert.parameters(), 'lr':self.bert_lr}, 
                                    {'params': self.model.Q.parameters(), 'lr':self.attention_lr}, 
                                    {'params': self.model.W.parameters(), 'lr':self.attention_lr}])
                return optimizer
            else:
                optimizer = Adam(self.parameters(), lr=self.learning_rate)
                if use_lr_scheduler:
                    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, linear_warmup_ratio, total_iters = 1)
                    linear_decay = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters = 9)
                    seq_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup, linear_decay], [1])
                    return [optimizer], [{"scheduler": seq_scheduler, "interval": "epoch"}]
                else:
                    return optimizer

if __name__ == '__main__':
    main()