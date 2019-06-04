
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

        # Convolutionals
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)

        # Fully connected
        self.fc1 = nn.Linear(16 * 9 * 36, 120)
        self.fc2 = nn.Linear(120, 84)

        # Batch norms
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)

        # Dropouts
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.batch_norm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 9 * 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DenseNet(nn.Module):
    def __init__(self, input_dim):
        super(DenseNet, self).__init__()

        # Fully connected
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)

        # Batch norms
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(128)

        # Dropouts
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        return x


class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        self.fc1 = nn.Linear(84 + 64, 128)
        self.fc2 = nn.Linear(128, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.fc = nn.Linear(21, 1)

    def forward(self, x_mat, x_com):
        x = torch.bmm(x_mat, x_com.unsqueeze(-1)).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x


class UnitedNet(nn.Module):
    def __init__(self, dense_dim, use_mord=True, use_mat=True, infer=False, dir_path=None, vis_thresh=0.8):
        super(UnitedNet, self).__init__()
        self.use_mord = use_mord
        self.use_mat = use_mat
        self.infer = infer
        self.vis_thresh = vis_thresh
        self.dir_path = dir_path
        if self.dir_path:
            self.smile_out_f = open(os.path.join(self.dir_path, 'smiles.txt'), 'w')
            self.weight_f = open(os.path.join(self.dir_path, 'weight.txt'), 'w')

        print('USING MAT = {}'.format(self.use_mat))

        # PARAMS FOR CNN NET
        # Convolutionals
        self.conv_conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv_pool = nn.MaxPool2d(2, 2)
        self.conv_conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.relu = nn.ReLU()

        # Fully connected
        self.conv_fc = nn.Linear(16 * 9 * 36, 120)

        # Batch norms
        self.conv_batch_norm1 = nn.BatchNorm2d(6)
        self.conv_batch_norm2 = nn.BatchNorm2d(16)

        # PARAMS FOR DENSE NET
        # Fully connected
        if self.use_mord:
            self.dense_fc1 = nn.Linear(dense_dim, 512)
            self.dense_fc2 = nn.Linear(512, 128)
            self.dense_fc3 = nn.Linear(128, 64)

            # Batch norms
            self.dense_batch_norm1 = nn.BatchNorm1d(512)
            self.dense_batch_norm2 = nn.BatchNorm1d(128)
            self.dense_batch_norm3 = nn.BatchNorm1d(64)

            # Dropouts
            self.dense_dropout = nn.Dropout()

        # PARAMS FOR COMBINED NET
        if self.use_mord:
            self.comb_fc = nn.Linear(120 + 64, 150)
        else:
            self.comb_fc = nn.Linear(120, 128)

        # PARAMS FOR ATTENTION NET
        if self.use_mat:
            self.att_fc = nn.Linear(42 + 150, 1)
        else:
            self.comb_fc_alt = nn.Linear(150, 1)

    def forward(self, x_non_mord, x_mord, x_mat, smiles=None):
        # FORWARD CNN
        x_non_mord = self.conv_conv1(x_non_mord)
        x_non_mord = self.conv_batch_norm1(x_non_mord)
        x_non_mord = self.relu(x_non_mord)
        x_non_mord = self.conv_pool(x_non_mord)

        x_non_mord = self.conv_conv2(x_non_mord)
        x_non_mord = self.conv_batch_norm2(x_non_mord)
        x_non_mord = self.relu(x_non_mord)
        x_non_mord = self.conv_pool(x_non_mord)

        x_non_mord = x_non_mord.view(x_non_mord.size(0), -1)
        x_non_mord = F.relu(self.conv_fc(x_non_mord))

        # FORWARD DENSE
        if self.use_mord:
            x_mord = self.relu(self.dense_fc1(x_mord))
            x_mord = self.dense_batch_norm1(x_mord)
            x_mord = self.dense_dropout(x_mord)

            x_mord = self.relu(self.dense_fc2(x_mord))
            x_mord = self.dense_batch_norm2(x_mord)
            x_mord = self.dense_dropout(x_mord)

            x_mord = self.relu(self.dense_fc3(x_mord))
            x_mord = self.dense_batch_norm3(x_mord)
            x_mord = self.dense_dropout(x_mord)

        # FORWARD COMBINE
        if self.use_mord:
            x_comb = torch.cat([x_non_mord, x_mord], dim=1)
        else:
            x_comb = x_non_mord

        x_comb = F.relu(self.comb_fc(x_comb))

        # FORWARD ATTENTION
        if self.use_mat:
            x_mat = torch.bmm(x_mat.permute(0, 2, 1), x_comb.unsqueeze(-1)).squeeze(-1)
            x_mat = torch.cat([x_mat, x_comb], dim=1)
            probs = torch.sigmoid(self.att_fc(x_mat))

            if self.infer:
                if not smiles:
                    raise ValueError('Please input smiles')
                alphas = x_comb.cpu().detach().numpy().tolist()
                alphas = ["\t".join([str(round(elem, 4)) for elem in seq]) for seq in alphas]
                prob_list = probs.cpu().detach().numpy().tolist()
                for smile, alpha, prob in zip(smiles, alphas, prob_list):
                    if prob[0] > self.vis_thresh:
                        self.weight_f.write(alpha + '\n')
                        self.smile_out_f.write(smile + '\n')

            return probs
        else:
            probs = torch.sigmoid(self.comb_fc_alt(x_comb))
            return probs

    def __del__(self):
        print('Closing files ...')
        if hasattr(self, 'weight_f'):
            self.weight_f.close()
        if hasattr(self, 'smile_out_f'):
            self.smile_out_f.close()


