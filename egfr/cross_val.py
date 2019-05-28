import argparse
import torch
import torch.nn as nn
import tensorboard_logger
from nets import UnitedNet
from torch.utils.data import dataloader
from dataset import EGFRDataset, train_cross_validation_split
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
                          lr,
                          fold):
    train_loader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         collate_fn=utils.custom_collate,
                                         shuffle=False)

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       collate_fn=utils.custom_collate,
                                       shuffle=False)

    # tensorboard_logger.configure('logs/' + hash_code)

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
        print('FOLD', fold, '-- EPOCH', e+1, '--', 'TRAINING')
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
        print('FOLD', fold, '-- EPOCH', e+1, '--', 'VALIDATION')
        for i, (mord_ft, non_mord_ft, label) in enumerate(val_loader):
            united_net.eval()
            mord_ft = mord_ft.float().to(val_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(val_device)
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device)
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
        if (e+1) % 10 == 0:
            print('{"metric": "train_loss", "value": %f, "epoch": %d}' % (sum(train_losses) / len(train_losses), e + 1))
            print('{"metric": "val_loss", "value": %f, "epoch": %d}' % (sum(val_losses) / len(val_losses), e + 1))
        for key in metrics.keys():
            train_metric = metrics[key](train_labels, train_outputs)
            val_metric = metrics[key](val_labels, val_outputs)
            if ((e+1) % 10) == 0:
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
            utils.save_model(united_net, "data/trained_models", hash_code + '_' + str(fold + 1))
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
    loader = dataloader.DataLoader(dataset=dataset,
                                   batch_size=128,
                                   collate_fn=utils.custom_collate,
                                   shuffle=True)
    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'), use_mat=True).to(device)
    united_net.load_state_dict(torch.load(model_path, map_location=device))
    # EVAL_MODE
    united_net.eval()
    out = []
    for i, (mord_ft, non_mord_ft, label) in enumerate(loader):
        with torch.no_grad():
            mord_ft = mord_ft.float().to(device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(device)
            mat_ft = non_mord_ft.squeeze(1).float().to(device)
            # Forward to get smiles and equivalent weights
            o = united_net(non_mord_ft, mord_ft, mat_ft)
            out.append(o)
    print('Forward done !!!')
    out = np.concatenate(out)
    out = out.squeeze()
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

    if args.gpu:
        train_device = 'cuda'
        val_device = 'cuda'
    else:
        train_device = 'cpu'
        val_device = 'cpu'

    tensorboard_logger.configure('logs/' + args.hashcode)

    train_metrics_cv = []
    val_metrics_cv = []
    best_cv = []
    metrics_dict = {'sensitivity': sensitivity,
                    'specificity': specificity,
                    'accuracy': accuracy,
                    'mcc': mcc, 'auc': auc}
    metrics_cv_dict = {'sensitivity': sensitivity_cv,
                       'specificity': specificity_cv,
                       'accuracy': accuracy_cv,
                       'mcc': mcc_cv,
                       'auc': auc_cv}

    for fold, (train_data, val_data) in enumerate(train_cross_validation_split(args.dataset)):
        train_dataset = EGFRDataset(train_data)
        val_dataset = EGFRDataset(val_data)

        train_metrics, val_metrics = train_validate_united(train_dataset,
                                                           val_dataset,
                                                           train_device,
                                                           val_device,
                                                           args.opt,
                                                           int(args.epochs),
                                                           int(args.batchsize),
                                                           {'sensitivity': sensitivity, 'specificity': specificity,
                                                            'accuracy': accuracy, 'mcc': mcc, 'auc': auc},
                                                           args.hashcode,
                                                           args.lr,
                                                           fold)

        train_metrics_cv.append(train_metrics)
        val_metrics_cv.append(val_metrics)
        filename = "data/trained_models/model_" + args.hashcode + '_' + str(fold + 1) + "_BEST"

        y_pred = predict(val_dataset, filename)
        y_true = val_dataset.label
        bestcv = []
        for m in metrics_cv_dict.values():
            bestcv.append(m(y_true, y_pred))

        print('val_metrics', val_metrics)
        print('Test', bestcv)
        best_cv.append(bestcv)

    train_metrics_cv = np.round(np.array([list(d.values()) for d in train_metrics_cv]).mean(axis=0), decimals=4)
    val_metrics_cv = np.round(np.array([list(d.values()) for d in val_metrics_cv]).mean(axis=0), decimals=4)
    best_cv = np.round(np.array([d for d in best_cv]).mean(axis=0), decimals=4)
    print(train_metrics_cv)
    print(val_metrics_cv)
    print(best_cv)
    row_format = "{:>12}" * (len(metrics_dict.keys()) + 1)
    print(row_format.format("", *(metrics_dict.keys())))
    print(row_format.format("Train", *train_metrics_cv))
    print(row_format.format("Val", *val_metrics_cv))
    print(row_format.format("Load", *best_cv))


if __name__ == '__main__':
    main()