
import torch
from torch.utils.data import dataloader
from torch.optim import SGD
from utils import create_dir


class BaseTrainer(object):

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 model,
                 params,
                 log_dir_path,
                 model_path,
                 checkpoint_path,
                 result_path,
                 logger,
                 prefix,
                 metrics):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.params = params

        # Check required attributes from params
        self.batch_size = self.params.get('batch_size', 128)
        self.n_epoch = self.params.get('n_epoch', 200)
        self.shuffle = self.params.get('shuffle', True)
        self.criterion = self.params.get('criterion', 'bce')
        self.lr = self.params.get('lr', 3e-4)
        self.mmt = self.params.get('mmt', .99)

        self.model = model
        self.opt = self.params.get('opt', SGD(self.model.parameters(),
                                              lr=self.lr, momentum=self.mmt))
        self.step = self.params.get('step', 100)
        self.logger = logger

        if self.criterion == 'nll':
            self.criterion = torch.nn.NLLLoss()
        elif self.criterion == 'bce':
            self.criterion = torch.nn.BCELoss()
        elif self.criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError('Unrecognized loss function !')

        # Metrics
        self.metrics = metrics

        # Data loader
        self.train_loader = dataloader.DataLoader(dataset=train_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        self.val_loader = dataloader.DataLoader(dataset=val_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)
        self.log_dir_path = log_dir_path
        self.model_path = model_path
        self.result_path = result_path
        self.checkpoint_path = checkpoint_path
        self.prefix = prefix

        # Create required paths
        create_dir(log_dir_path)
        create_dir(model_path)
        create_dir(checkpoint_path)

    def train(self, model, n_epoch, train_device, val_device):
        for e in range(n_epoch):
            print('EPOCH [{}]'.format(e + 1))
            self.forward(self.train_loader, model, True, e, train_device)
            _, _, loss = self.run(model, "{}/model_{}".format(self.model_path, e + 1), val_device)
            self.save_log(e, 'val_loss', loss)

    def run(self, model, model_path, run_device):
        model = self.load_model(model, model_path, run_device)
        uids, outputs, loss = self.forward(self.val_loader, model, False, 1, run_device)
        return uids, outputs, loss

    def load_model(self, model, model_path, run_device):
        with torch.no_grad():
            model.load_state_dict(torch.load(model_path, map_location=run_device))
            model.eval()
        return model

    def save_model(self, e):
        torch.save(self.model.state_dict(), "{}/model_{}".format(self.model_path, e + 1))
        state = {'epoch': e, 'state_dict': self.model.state_dict(), 'optimizer': self.opt.state_dict()}
        torch.save(state, "{}/state_{}".format(self.checkpoint_path, e + 1))

    def save_log(self, e, name, value):
        self.logger.log_value(name, value, e)

    def forward(self, loader, model, with_grad, run_step, run_device):
        raise NotImplementedError

