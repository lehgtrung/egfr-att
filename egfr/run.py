import argparse
import torch
import torch.nn as nn
import tensorboard_logger
from nets import CnnNet, DenseNet, CombinedNet, AttentionNet, UnitedNet
from torch.utils.data import dataloader
from dataset import EGFRDataset, train_validation_split
import torch.optim as optim
from metrics import *
import collections


def get_max_length(x):
    return len(max(x, key=len))


def pad_sequence(seq):
    def _pad(_it, _max_len):
        return [0] * (_max_len - len(_it)) + _it
    padded = [_pad(it, get_max_length(seq)) for it in seq]
    return padded


def custom_collate(batch):
    """
        Custom collate function for our batch, a batch in dataloader looks like
            [(0, [24104, 27359], 6684),
            (0, [24104], 27359),
            (1, [16742, 31529], 31485),
            (1, [16742], 31529),
            (2, [6579, 19316, 13091, 7181, 6579, 19316], 13091)]
    """
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        if isinstance(samples[0], int):
            lst.append(torch.LongTensor(samples))
        elif isinstance(samples[0], float):
            lst.append(torch.DoubleTensor(samples))
        elif isinstance(samples[0], collections.Sequence):
            lst.append(torch.LongTensor(pad_sequence(samples)))
    return lst


def train_validate(train_dataset,
                   val_dataset,
                   train_device,
                   val_device,
                   opt_type,
                   n_epoch,
                   batch_size,
                   metrics,
                   hash_code,
                   lr):
    train_loader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         collate_fn=custom_collate,
                                         shuffle=True)

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       collate_fn=custom_collate,
                                       shuffle=True)

    tensorboard_logger.configure('logs/' + hash_code)

    criterion = nn.BCELoss()
    cnn_net = CnnNet().to(train_device)
    dense_net = DenseNet(input_dim=train_dataset.get_dim('mord')).to(train_device)
    combined_net = CombinedNet().to(train_device)
    attention_net = AttentionNet().to(train_device)

    if opt_type == 'sgd':
        opt = optim.SGD(list(cnn_net.parameters()) + list(dense_net.parameters()) + list(combined_net.parameters()),
                        lr=lr,
                        momentum=0.99)
    elif opt_type == 'adam':
        opt = optim.Adam(list(cnn_net.parameters()) + list(dense_net.parameters()) + list(combined_net.parameters()),
                         lr=lr)

    for e in range(n_epoch):
        train_losses = []
        val_losses = []
        train_outputs = []
        val_outputs = []
        train_labels = []
        val_labels = []
        print(e, '--', 'TRAINING ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(train_loader):
            mord_ft = mord_ft.float().to(train_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 42, 150)).float().to(train_device)
            mat_ft = non_mord_ft.narrow(2, 0, 21).view((-1, 21, 150)).float().to(train_device)
            label = label.float().to(train_device)

            # Forward
            opt.zero_grad()
            non_mord_output = cnn_net(non_mord_ft)
            mord_output = dense_net(mord_ft)

            combined_output = combined_net(mord_output, non_mord_output)
            # print('Combined net shape: ', combined_output.shape)
            # print('Mat ft shape: ', mat_ft.shape)
            outputs = attention_net(mat_ft, combined_output)

            loss = criterion(outputs, label)
            train_losses.append(float(loss.item()))
            train_outputs.extend(outputs)
            train_labels.extend(label)

            # Parameters update
            loss.backward()
            opt.step()

        # Validate after each epoch
        print(e, '--', 'VALIDATION ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(val_loader):
            mord_ft = mord_ft.float().to(val_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 42, 150)).float().to(val_device)
            mat_ft = non_mord_ft.narrow(2, 0, 21).view((-1, 21, 150)).float().to(val_device)
            label = label.float().to(val_device)

            with torch.no_grad():
                non_mord_output = cnn_net(non_mord_ft)
                mord_output = dense_net(mord_ft)

                combined_output = combined_net(mord_output, non_mord_output)
                # print('Combined net shape: ', combined_output.shape)
                # print('Mat ft shape: ', mat_ft.shape)
                outputs = attention_net(mat_ft, combined_output)

                loss = criterion(outputs, label)
                val_losses.append(float(loss.item()))
                val_outputs.extend(outputs)
                val_labels.extend(label)

        train_outputs = torch.stack(train_outputs)
        val_outputs = torch.stack(val_outputs)
        train_labels = torch.stack(train_labels)
        val_labels = torch.stack(val_labels)
        tensorboard_logger.log_value('train_loss', sum(train_losses) / len(train_losses), e + 1)
        tensorboard_logger.log_value('val_loss', sum(val_losses) / len(val_losses), e + 1)

        for key in metrics.keys():
            train_metric = metrics[key](train_labels, train_outputs)
            val_metric = metrics[key](val_labels, val_outputs)
            tensorboard_logger.log_value('train_{}'.format(key),
                                         train_metric, e + 1)
            tensorboard_logger.log_value('val_{}'.format(key),
                                         val_metric, e + 1)

    train_metrics = {}
    val_metrics = {}
    for key in metrics.keys():
        train_metrics[key] = metrics[key](train_labels, train_outputs)
        val_metrics[key] = metrics[key](val_labels, val_outputs)

    return train_metrics, val_metrics


def train_validate_united(train_dataset,
                          val_dataset,
                          train_device,
                          val_device,
                          opt_type,
                          n_epoch,
                          batch_size,
                          metrics,
                          hash_code,
                          lr):
    train_loader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         collate_fn=custom_collate,
                                         shuffle=True)

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       collate_fn=custom_collate,
                                       shuffle=True)

    tensorboard_logger.configure('logs/' + hash_code)

    criterion = nn.BCELoss()
    united_net = UnitedNet(dense_dim=train_dataset.get_dim('mord'), use_mat=True).to(train_device)

    if opt_type == 'sgd':
        opt = optim.SGD(united_net.parameters(),
                        lr=lr,
                        momentum=0.99)
    elif opt_type == 'adam':
        opt = optim.Adam(united_net.parameters(),
                         lr=lr)

    for e in range(n_epoch):
        train_losses = []
        val_losses = []
        train_outputs = []
        val_outputs = []
        train_labels = []
        val_labels = []
        print(e, '--', 'TRAINING ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(train_loader):
            mord_ft = mord_ft.float().to(train_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 42, 150)).float().to(train_device)
            mat_ft = non_mord_ft.narrow(2, 0, 21).view((-1, 21, 150)).float().to(train_device)
            label = label.float().to(train_device)

            # Forward
            opt.zero_grad()
            outputs = united_net(non_mord_ft, mord_ft, mat_ft)

            loss = criterion(outputs, label)
            train_losses.append(float(loss.item()))
            train_outputs.extend(outputs)
            train_labels.extend(label)

            # Parameters update
            loss.backward()
            opt.step()

        # Validate after each epoch
        print(e, '--', 'VALIDATION ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(val_loader):
            mord_ft = mord_ft.float().to(val_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 42, 150)).float().to(val_device)
            mat_ft = non_mord_ft.narrow(2, 0, 21).view((-1, 21, 150)).float().to(val_device)
            label = label.float().to(val_device)

            with torch.no_grad():
                outputs = united_net(non_mord_ft, mord_ft, mat_ft)

                loss = criterion(outputs, label)
                val_losses.append(float(loss.item()))
                val_outputs.extend(outputs)
                val_labels.extend(label)

        train_outputs = torch.stack(train_outputs)
        val_outputs = torch.stack(val_outputs)
        train_labels = torch.stack(train_labels)
        val_labels = torch.stack(val_labels)
        tensorboard_logger.log_value('train_loss', sum(train_losses) / len(train_losses), e + 1)
        tensorboard_logger.log_value('val_loss', sum(val_losses) / len(val_losses), e + 1)

        for key in metrics.keys():
            train_metric = metrics[key](train_labels, train_outputs)
            val_metric = metrics[key](val_labels, val_outputs)
            tensorboard_logger.log_value('train_{}'.format(key),
                                         train_metric, e + 1)
            tensorboard_logger.log_value('val_{}'.format(key),
                                         val_metric, e + 1)

    train_metrics = {}
    val_metrics = {}
    for key in metrics.keys():
        train_metrics[key] = metrics[key](train_labels, train_outputs)
        val_metrics[key] = metrics[key](val_labels, val_outputs)

    return train_metrics, val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset', dest='dataset', default='data/egfr_10_full_ft_pd_lines.json')
    parser.add_argument('-e', '--epochs', help='Number of epochs', dest='epochs', default=500)
    parser.add_argument('-b', '--batchsize', help='Batch size', dest='batchsize', default=128)
    parser.add_argument('-o', '--opt', help='Optimizer adam or sgd', dest='opt', default='adam')
    parser.add_argument('-g', '--gpu', help='Use GPU or Not?', action='store_true')
    parser.add_argument('-c', '--hashcode', help='Hashcode for tf.events', dest='hashcode', default='TEST')
    parser.add_argument('-l', '--lr', help='Learning rate', dest='lr', default=1e-5, type=float)
    args = parser.parse_args()

    train_data, val_data = train_validation_split(args.dataset)
    train_dataset = EGFRDataset(train_data)
    val_dataset = EGFRDataset(val_data)
    if args.gpu:
        train_device='cuda'
        val_device='cuda'
    else:
        train_device = 'cpu'
        val_device = 'cpu'
    train_validate_united(train_dataset,
                          val_dataset,
                          train_device,
                          val_device,
                          args.opt,
                          int(args.epochs),
                          int(args.batchsize),
                          {'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy, 'mcc': mcc, 'auc': auc},
                          args.hashcode,
                          args.lr)


if __name__ == '__main__':
    main()




