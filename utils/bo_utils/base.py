import torch
from collections import OrderedDict


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish):
        if swish:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("swish", Swish()),
            ]))
        else:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("norm", torch.nn.BatchNorm1d(output_dim)),
                ("relu", torch.nn.ReLU(True)),
            ]))


class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims, swish=True):
        if isinstance(hidden_dims, str):
            model, output_dim = image_classifier(hidden_dims)
            layers = model
            self.output_dim = output_dim
        else:
            prev_dims = [input_dim] + list(hidden_dims[:-1])
            layers = OrderedDict([
                (f"hidden{i + 1}", _LinearBlock(prev_dim, current_dim, swish=swish))
                for i, (prev_dim, current_dim) in enumerate(zip(prev_dims, hidden_dims))
            ])
            self.output_dim = hidden_dims[-1]

        super().__init__(layers)