import os
import torch
from torch import nn

class StreamingLDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant
    Analysis algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, backbone=None, shrinkage_param=1e-4, streaming_update_sigma=True, ood_type='mahalanobis', device='cuda'):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        """

        super(StreamingLDA, self).__init__()

        # SLDA parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma
        self.ood_type = ood_type

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)  # covariance
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the SLDA model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # covariance updates
        if self.streaming_update_sigma:
            x_minus_mu = (x - self.muK[y])
            mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates / (self.num_updates + 1)
            self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def predict(self, X):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        # compute/load Lambda matrix
        if self.prev_num_updates != self.num_updates:
            # there have been updates to the model, compute Lambda
            Lambda = torch.pinverse(
                (1 - self.shrinkage_param) * self.Sigma
                + self.shrinkage_param 
                * torch.eye(self.input_shape).to(self.device)
            )
            self.Lambda = Lambda
            self.prev_num_updates = self.num_updates

        # parameters for predictions
        M = self.muK.transpose(1, 0)
        W = torch.matmul(self.Lambda, M)
        c = 0.5 * torch.sum(M * W, dim=0)

        # loop in mini-batches over test samples
        scores = torch.matmul(X, W) - c

        return scores

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y):
        # fit SLDA one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), None)

    @torch.no_grad()
    def train_(self, feature, target):
        if self.backbone is not None:
            batch_x_feat = self.backbone(feature.to(self.device))
        else:
            batch_x_feat = feature.to(self.device)

        self.fit_batch(batch_x_feat, target)

    @torch.no_grad()
    def evaluate_(self, feature):
        probas = self.predict(feature)

        return probas

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Sigma'] = self.Sigma.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_file))
        print('\nloading ckpt from: %s' % save_file)
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Sigma = d['Sigma'].to(self.device)
        self.num_updates = d['num_updates']
