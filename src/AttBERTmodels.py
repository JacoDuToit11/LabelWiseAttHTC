from transformers import AutoModel
import torch
import torch.nn as nn
from torch.nn import functional as F

class FlatBERTAttModel(nn.Module):
    def __init__(self, num_labels, model_name, concat_last_4_hidden_states, dropout_prob):
        super(FlatBERTAttModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.model_type = self.bert.config.model_type
        hidden_size = self.bert.config.hidden_size
        self.concat_last_4_hidden_states = concat_last_4_hidden_states

        if concat_last_4_hidden_states:
            self.Q = nn.Linear(hidden_size * 4, num_labels, bias=False)
            self.W = nn.Linear(hidden_size * 4, num_labels)
        else:
            self.Q = nn.Linear(hidden_size, num_labels, bias=False)
            self.W = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        if self.concat_last_4_hidden_states:
            if self.model_type == 'bert' or self.model_type == 'roberta':
                last_hidden_states, CLS, all_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                                                                       output_hidden_states=True, return_dict=False)
            else:
                # TODO not sure if this works
                hidden_states, all_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=False)
            hidden_states = []
            num_layers = len(all_hidden_states)
            if num_layers - 4 <= 0:
                start = 1
            else:
                start = num_layers - 4
            for i in range(start, num_layers):
                hidden_states.append(all_hidden_states[i])
            hidden_states = torch.cat(hidden_states, dim = 2)
        else:
            if self.model_type == 'bert' or self.model_type == 'roberta':
                hidden_states, CLS = self.bert(input_ids, attention_mask, return_dict = False)
            else:
                hidden_states = self.bert(input_ids, attention_mask, return_dict = False)[0]
        
        weights = self.Q(hidden_states)
        att_weights = F.softmax(weights, 1).transpose(1, 2)
        weighted_output = att_weights @ hidden_states
        weighted_output = self.W.weight.mul(weighted_output).sum(dim=2).add(self.W.bias)
        return weighted_output
    
class FlatBERTLaatModel(nn.Module):
    def __init__(self, num_labels, model_name, concat_last_4_hidden_states, laat_projection_dim, dropout_prob):
        super(FlatBERTLaatModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.concat_last_4_hidden_states = concat_last_4_hidden_states
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p = dropout_prob)

        projection_dim = laat_projection_dim
        self.model_type = self.bert.config.model_type
        hidden_size = self.bert.config.hidden_size
        if concat_last_4_hidden_states:
            self.W = nn.Linear(hidden_size * 4, projection_dim, bias = False)
            self.U = nn.Linear(projection_dim, num_labels, bias = False)
            self.final_linear = nn.Linear(hidden_size * 4, num_labels)
        else:
            self.W = nn.Linear(hidden_size, projection_dim, bias = False)
            self.U = nn.Linear(projection_dim, num_labels, bias = False)
            self.final_linear = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        if self.concat_last_4_hidden_states:
            if self.model_type == 'bert' or self.model_type == 'roberta':
                last_hidden_states, CLS, all_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                                                                       output_hidden_states=True, return_dict=False)
            else:
                hidden_states, all_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                                                             output_hidden_states=True, return_dict=False)
            hidden_states = []
            num_layers = len(all_hidden_states)
            if num_layers - 4 <= 0:
                start = 1
            else:
                start = num_layers - 4
            for i in range(start, num_layers):
                hidden_states.append(all_hidden_states[i])
            hidden_states = torch.cat(hidden_states, dim = 2)
        else:
            if self.model_type == 'bert' or self.model_type == 'roberta':
                hidden_states, CLS = self.bert(input_ids, attention_mask, return_dict = False)
            else:
                hidden_states = self.bert(input_ids, attention_mask, return_dict = False)[0]

        weights = F.tanh(self.W(hidden_states))

        att_weights = self.U(weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        weighted_output = att_weights @ hidden_states
        if self.dropout_prob > 0:
            weighted_output = self.dropout(weighted_output)
        weighted_output = self.final_linear.weight.mul(weighted_output).sum(dim=2).add(
                self.final_linear.bias)
        return weighted_output

class JointLaatModel(nn.Module):
    def __init__(self, num_labels, level_num_labels, depth2label, model_name, concat_last_4_hidden_states, laat_projection_dim, 
                 level_projection_dim, concat_all_previous_levels, use_previous_level_output_scores):
        super(JointLaatModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.num_labels = num_labels
        self.level_num_labels = level_num_labels
        self.num_levels = len(level_num_labels)
        self.hidden_size = self.bert.config.hidden_size
        self.model_type = self.bert.config.model_type
        self.depth2label = depth2label
        self.concat_all_previous_levels = concat_all_previous_levels
        self.use_previous_level_output_scores = use_previous_level_output_scores

        self.attention_model = JointLaatAttention(self.hidden_size, level_num_labels, concat_last_4_hidden_states, laat_projection_dim, 
                                                  level_projection_dim, concat_all_previous_levels, use_previous_level_output_scores)
        
        if not self.use_previous_level_output_scores:
            self.projection_linears = nn.ModuleList([nn.Linear(level_num_labels[level], level_projection_dim, bias = False) for level in range(self.num_levels)])

    def forward(self, input_ids, attention_mask, previous_level_projection=None, level=0):
        if self.model_type == 'bert' or self.model_type == 'roberta':
            hidden_states, CLS = self.bert(input_ids, attention_mask, return_dict = False)
        else:
            hidden_states = self.bert(input_ids, attention_mask, return_dict = False)[0]

        previous_level_projection = None
        weighted_output = torch.zeros(input_ids.shape[0], self.num_labels).to(input_ids.device)
        for level in range(self.num_levels):
            level_output = self.attention_model(hidden_states, previous_level_projection, level)
            if self.concat_all_previous_levels:
                if self.use_previous_level_output_scores:
                    temp_previous_level_projection = torch.sigmoid(level_output)
                else:
                    temp_previous_level_projection = self.projection_linears[level](torch.sigmoid(level_output))
                    temp_previous_level_projection = torch.sigmoid(temp_previous_level_projection)
                if level == 0:
                    previous_level_projection = temp_previous_level_projection
                else:
                    previous_level_projection = torch.cat([previous_level_projection, temp_previous_level_projection], dim = 1)
            else:
                if self.use_previous_level_output_scores:
                    previous_level_projection = torch.sigmoid(level_output)
                else:
                    previous_level_projection = self.projection_linears[level](torch.sigmoid(level_output))
                    previous_level_projection = torch.sigmoid(previous_level_projection)
            level_indices = self.depth2label[level]
            weighted_output[:, level_indices] = level_output
        return weighted_output
    
class JointLaatAttention(nn.Module):
    def __init__(self, hidden_size, level_num_labels, concat_last_4_hidden_states, projection_dim, level_projection_dim, 
                 concat_all_previous_levels, use_previous_level_output_scores):
        super(JointLaatAttention, self).__init__()
        self.level_num_labels = level_num_labels
        self.concat_all_previous_levels = concat_all_previous_levels
        self.use_previous_level_output_scores = use_previous_level_output_scores
        num_levels = len(level_num_labels)

        self.W = nn.ModuleList([nn.Linear(hidden_size, projection_dim, bias = False) for level in range(num_levels)])
        self.U = nn.ModuleList([nn.Linear(projection_dim, level_num_labels[level], bias = False) for level in range(num_levels)])

        if self.concat_all_previous_levels:
            if self.use_previous_level_output_scores:
                summed_previous_levels_labels = []
                total_labels = 0
                for level in range(num_levels):
                    summed_previous_levels_labels.append(total_labels)
                    total_labels += level_num_labels[level]
                self.final_linear = nn.ModuleList([nn.Linear(hidden_size +
                                                                summed_previous_levels_labels[level],
                                                                level_num_labels[level], bias = True) for level in range(num_levels)])
            else:
                self.final_linear = nn.ModuleList([nn.Linear(hidden_size + 
                                                                (level_projection_dim * level),
                                                                level_num_labels[level], bias = True) for level in range(num_levels)])
        else:
            if self.use_previous_level_output_scores:
                self.final_linear = nn.ModuleList([nn.Linear(hidden_size + 
                                                                (level_num_labels[level - 1] if level > 0 else 0),
                                                                level_num_labels[level], bias = True) for level in range(num_levels)])
            else:
                self.final_linear = nn.ModuleList([nn.Linear(hidden_size + 
                                                                (level_projection_dim if level > 0 else 0),
                                                                level_num_labels[level], bias = True) for level in range(num_levels)])

    def forward(self, hidden_states, previous_level_projection=None, level=0):
        weights = F.tanh(self.W[level](hidden_states))
        att_weights = self.U[level](weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        weighted_output = att_weights @ hidden_states

        batch_size = weighted_output.shape[0]
        if previous_level_projection is not None:
            concatted_output = [weighted_output,
                                previous_level_projection.repeat(1, self.level_num_labels[level]).view(batch_size, self.level_num_labels[level], -1)]
            weighted_output = torch.cat(concatted_output, dim = 2)

        weighted_output = self.final_linear[level].weight.mul(weighted_output).sum(dim=2).add(
                self.final_linear[level].bias)
        return weighted_output
    
class HierAttBertClassifier(nn.Module):
    def __init__(self, hier_dict, num_labels, model_type, model_name, laat_projection_dim, dropout_prob):
        super(HierAttBertClassifier, self).__init__()
        self.hier_dict = hier_dict
        self.num_labels = num_labels
        
        if model_type == 'FlatBERTLaatModel':
            self.model_list = nn.ModuleList([FlatBERTLaatModel(len(hier_dict), model_name, laat_projection_dim)])
            # Level 1 classifiers
            self.model_list.extend([FlatBERTLaatModel(len(hier_dict[i]), model_name) for i in range(len(hier_dict))])
        elif model_type == 'FlatBERTAttModel':
            self.model_list = nn.ModuleList([FlatBERTAttModel(len(hier_dict), model_name)])
            # Level 1 classifiers
            self.model_list.extend([FlatBERTAttModel(len(hier_dict[i]), model_name) for i in range(len(hier_dict))])
    
    def forward(self, input_ids, attention_mask):
        # Pass embeddings to CNN
        output = torch.zeros(input_ids.shape[0], self.num_labels)
        root_classifier_outputs = self.model_list[0](input_ids, attention_mask)
        level_1_predictions = torch.argmax(root_classifier_outputs, dim = 1).tolist()
        
        for i in range(output.shape[0]):
            level_1_classifier_output = self.model_list[1 + level_1_predictions[i]](torch.unsqueeze(input_ids[i], attention_mask[i], dim = 0))
            level_2_prediction = torch.argmax(level_1_classifier_output)
            level_2_prediction_global = self.hier_dict[level_1_predictions[i]][level_2_prediction]
            output[i][level_1_predictions[i]] = 1
            output[i][level_2_prediction_global] = 1
        return output