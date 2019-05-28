import argparse
import torch
import torch.nn as nn
import tensorboard_logger
from nets import UnitedNet
from torch.utils.data import dataloader
from dataset import EGFRDataset, train_validation_split
import torch.optim as optim
from metrics import *
import utils


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
                                         collate_fn=utils.custom_collate,
                                         shuffle=False)

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       collate_fn=utils.custom_collate,
                                       shuffle=False)

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

    min_loss = 100  # arbitary large number
    min_loss_idx = 0
    early_stop_count = 0
    for e in range(n_epoch):
        train_losses = []
        val_losses = []
        train_outputs = []
        val_outputs = []
        train_labels = []
        val_labels = []
        print(e, '--', 'TRAINING ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(train_loader):
            united_net.train()
            mord_ft = mord_ft.float().to(train_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(train_device)
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device)
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
            united_net.eval()
            mord_ft = mord_ft.float().to(val_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(val_device)
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device)
            label = label.float().to(val_device)

            with torch.no_grad():
                outputs = united_net(non_mord_ft, mord_ft, mat_ft)
                # print(outputs[-1, -20:])

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
        print('{"metric": "train_loss", "value": %f, "epoch": %d}' % (sum(train_losses) / len(train_losses), e + 1))
        print('{"metric": "val_loss", "value": %f, "epoch": %d}' % (sum(val_losses) / len(val_losses), e + 1))
        for key in metrics.keys():
            train_metric = metrics[key](train_labels, train_outputs)
            val_metric = metrics[key](val_labels, val_outputs)
            print('{"metric": "%s", "value": %f, "epoch": %d}' % ('train_' + key, train_metric, e + 1))
            print('{"metric": "%s", "value": %f, "epoch": %d}' % ('val_' + key, val_metric, e + 1))
            tensorboard_logger.log_value('train_{}'.format(key),
                                         train_metric, e + 1)
            tensorboard_logger.log_value('val_{}'.format(key),
                                         val_metric, e + 1)
        loss_epoch = sum(val_losses) / len(val_losses)
        if loss_epoch < min_loss:
            early_stop_count = 0
            min_loss_idx = e
            print(min_loss_idx)
            min_loss = loss_epoch
            utils.save_model(united_net, "data/trained_models", hash_code)
        else:
            early_stop_count += 1
            if early_stop_count > 20:
                print('Traning can not improve from epoch {}\tBest loss: {}'.format(e, min_loss))
                break

    train_metrics = {}
    val_metrics = {}
    for key in metrics.keys():
        train_metrics[key] = metrics[key](train_labels, train_outputs)
        val_metrics[key] = metrics[key](val_labels, val_outputs)

    return train_metrics, val_metrics


def predict(dataset, model_path, device='cpu'):
    losses = []
    criterion = nn.BCELoss()
    loader = dataloader.DataLoader(dataset=dataset,
                                   batch_size=128,
                                   collate_fn=utils.custom_collate,
                                   shuffle=False)
    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'), use_mat=True).to(device)
    united_net.load_state_dict(torch.load(model_path, map_location=device))
    united_net.eval()
    out = []
    for i, (mord_ft, non_mord_ft, label) in enumerate(loader):
        with torch.no_grad():
            mord_ft = mord_ft.float().to(device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(device)
            mat_ft = non_mord_ft.squeeze(1).float().to(device)
            label = label.float().to(device)
            # Forward to get smiles and equivalent weights
            o = united_net(non_mord_ft, mord_ft, mat_ft)
            losses.append(criterion(o, label).item())
            out.append(o)
    print('Forward done !!!')
    print('Loss prediction: ', sum(losses) / len(losses))
    out = np.concatenate(out)
    print(out[np.where(out > 0.5)])
    print(len(out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset', dest='dataset',
                        default='data/egfr_10_full_ft_pd_lines.json')
    parser.add_argument('-e', '--epochs', help='Number of epochs', dest='epochs', default=500)
    parser.add_argument('-b', '--batchsize', help='Batch size', dest='batchsize', default=128)
    parser.add_argument('-o', '--opt', help='Optimizer adam or sgd', dest='opt', default='adam')
    parser.add_argument('-g', '--gpu', help='Use GPU or Not?', action='store_true')
    parser.add_argument('-c', '--hashcode', help='Hashcode for tf.events', dest='hashcode', default='TEST')
    parser.add_argument('-l', '--lr', help='Learning rate', dest='lr', default=1e-5, type=float)
    parser.add_argument('-k', '--mode', help='Train or predict ?', dest='mode', default='train', type=str)
    parser.add_argument('-m', '--model_path', help='Trained model path', dest='model_path', type=str)
    args = parser.parse_args()

    train_data, val_data = train_validation_split(args.dataset)
    train_dataset = EGFRDataset(train_data)
    val_dataset = EGFRDataset(val_data)
    if args.gpu:
        train_device = 'cuda'
        val_device = 'cuda'
    else:
        train_device = 'cpu'
        val_device = 'cpu'

    if args.mode == 'train':
        train_validate_united(train_dataset,
                              val_dataset,
                              train_device,
                              val_device,
                              args.opt,
                              int(args.epochs),
                              int(args.batchsize),
                              {'sensitivity': sensitivity, 'specificity': specificity,
                               'accuracy': accuracy, 'mcc': mcc, 'auc': auc},
                              args.hashcode,
                              args.lr)
    else:
        pred_dataset = EGFRDataset(args.dataset)
        predict(pred_dataset, args.model_path, train_device)


if __name__ == '__main__':
    main()


