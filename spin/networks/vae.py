import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(
        self,
        n_components=None,
        learning_rate=0.001,
        batch_size=64,
        n_iter=100,
        verbose=False,
    ):

        super(VAE, self).__init__()

        self.n_hidden = n_components

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        # this is a bit awkward IMO, but trying to match sklearn's API,
        # so we set visible according to the data input to the fit method
        self.n_visible = None
        self.encoder = None
        self.decoder = None

        self.verbose = verbose

    def fit(self, training_data):
        self.n_visible = training_data[0].size
        self.encoder = nn.Sequential(nn.Linear(self.n_visible, self.n_hidden))
        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_visible), nn.Sigmoid()
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        compute_loss = nn.MSELoss()

        train_data = torch.from_numpy(training_data).float()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.batch_size)

        for epoch in range(self.n_iter):
            for step, x in enumerate(train_loader):
                b_x = Variable(x.view(-1, self.n_visible))

                encoded = self.encoder(b_x)
                decoded = self.decoder(encoded)
                error = compute_loss(decoded, b_x)

                if self.verbose and step == 0:
                    print(epoch, error)

                optimizer.zero_grad()
                error.backward()
                optimizer.step()

                """
                if self.n_log > 0:
                    if epoch + step == 0:
                        train_log = {
                            'n_log': self.n_log,
                            'epoch': [],
                            'error': [],
                            'references': [],
                            'reconstructed': []
                        }
                        
                    if step == 0:
                        train_log['epoch'].append(epoch)
                        train_log['error'].append(error)
                            if len(train_log['references']) == 0:
                                references = []
                                for example in range(self.n_log):
                                    references.append(train_data[example:example+1])
                                train_log['references'] = references
    
                            log = []
                            for example in range(self.n_log):
                                log.append(decoded.data.numpy()[example:example+1])
                            train_log['reconstructed'].append(log)
                """

        self.score = error.item()

        return self

    def score_samples(self, data):
        x = torch.from_numpy(data).float()

        b_x = Variable(x.view(-1, self.n_visible))
        encoded = self.encoder(b_x)
        decoded = self.decoder(encoded)

        compute_loss = nn.MSELoss()
        error = compute_loss(decoded, b_x)

        return [error.item()]
