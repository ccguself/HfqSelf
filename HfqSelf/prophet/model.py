import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, conv_list ,device):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.conv_list = conv_list
        self.device = device
        input_dim_comb = input_dim * in_seq_length
        hidden_layer11 = nn.Linear(input_dim_comb, hidden_dim)
        hidden_layer12 = [nn.Linear(input_dim_comb + hidden_dim + output_dim, hidden_dim) for _ in range(len(conv_list))]
        hidden_layer1_list = [[hidden_layer11] for _ in range(len(conv_list))]
        for k in range(len(conv_list)): 
            for i in range(out_seq_length//conv_list[k] - 1):
                hidden_layer1_list[k].append(hidden_layer12[k])
            hidden_layer1_list[k] = nn.ModuleList( hidden_layer1_list[k])
        self.hidden_layer1_list = nn.ModuleList(hidden_layer1_list)
        self.hidden_layer2_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(len(conv_list))])
        self.output_layer_list = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(len(conv_list))])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

    def forward(self, input, target, is_training=False):
        output_list = []
        t = -1
        for k in range(len(self.conv_list)):
            outputs = torch.zeros((self.out_seq_length//self.conv_list[k], input.shape[0], self.output_dim)).to(self.device)
            next_cell_input = input
            for i in range(self.out_seq_length//self.conv_list[k]):
                hidden = F.relu(self.hidden_layer1_list[k][i](next_cell_input))
                hidden = F.relu(self.hidden_layer2_list[k](hidden))
                output = self.output_layer_list[k](hidden)
                outputs[i,:,:] = output
                if is_training:
                    t = t + 1
                    next_cell_input = torch.cat((input, hidden, output), dim=1)
                else:
                    next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1)
            output_list.append(outputs)
        return torch.cat(output_list,dim=0)
    
    
class FCN_1(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, device):
        super(FCN_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device
        input_dim_comb = input_dim * in_seq_length
        hidden_layer1 = [nn.Linear(input_dim_comb, hidden_dim)]
        for i in range(out_seq_length - 1):
            hidden_layer1.append(nn.Linear(input_dim_comb + hidden_dim + output_dim, hidden_dim))
        self.hidden_layer1 = nn.ModuleList(hidden_layer1)
        self.hidden_layer2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(out_seq_length)])
        self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
    def forward(self, input, target, is_training=False):
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        next_cell_input = input
        for i in range(self.out_seq_length):
            hidden = F.relu(self.hidden_layer1[i](next_cell_input))
            hidden = F.relu(self.hidden_layer2[i](hidden))
            output = self.output_layer[i](hidden)
            outputs[i,:,:] = output
            if is_training:
                next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1)
            else:
                next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1)
        return outputs    