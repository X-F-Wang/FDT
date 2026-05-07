import torch.nn as nn
import torch
from models import register
import numpy as np
import torch.nn.functional as F
#This is a completely different version from FAN.
# For simplicity, I haven't changed the module name.

class FANLayer(nn.Module):

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super().__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cos and sin

        # Linear transformation for p component (for cos/sin)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        # Linear transformation for g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        return torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)

"""
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.25,
                 activation='gelu', use_p_bias=True, omega=10.0):
        super().__init__()
        assert 0 < p_ratio < 0.5

        self.omega = omega
        p_output_dim = int(output_dim * p_ratio)
        g_output_dim = output_dim - 2 * p_output_dim

        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src) * self.omega
        return torch.cat((g, torch.cos(p), torch.sin(p)), dim=-1)
"""

@register('fan')
class FAN_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list,
                 p_ratio=0.25, act='gelu', use_p_bias=True):
        super().__init__()

        # Handle activation specification
        if act is None:
            activation_func = None
        elif act.lower() == 'relu':
            activation_func = 'relu'
        elif act.lower() == 'gelu':
            activation_func = 'gelu'
        else:
            assert False, f'activation {act} is not supported'

        layers = []
        lastv = in_dim

        # Build FAN hidden layers
        for hidden in hidden_list:
            layers.append(FANLayer(
                lastv,
                hidden,
                p_ratio=p_ratio,
                activation=activation_func,
                use_p_bias=use_p_bias
            ))
            lastv = hidden

        # Final output layer (regular linear layer)
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # Create and test the model
    model = FAN_MLP(
        in_dim=256,
        out_dim=3,
        hidden_list=[256, 256, 256, 256],
        p_ratio=0.25,
        act='gelu',
        use_p_bias=True
    )

    # Print model structure and parameter count
    print(model)
    print(f"Total parameters: {count_parameters(model)}")

    # Test forward pass
    test_input = torch.randn(2, 5, 256)  # batch=2, seq_len=5, features=256
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")