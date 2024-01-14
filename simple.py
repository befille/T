import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.init as init
import matplotlib.pyplot as plt
import datasets
from einops import einsum, rearrange, repeat


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
    
class MambaBlock(nn.Module):
    """
    "A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."
    """
    def __init__(
        self,
        dim: int = None,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # If dt_rank is not provided, set it to ceil(dim / d_state)
        dt_rank = math.ceil(self.dim / 16)
        self.dt_rank = dt_rank

        # If dim_inner is not provided, set it to dim * expand
        dim_inner = dim * expand
        self.dim_inner = dim_inner

        # If dim_inner is not provided, set it to dim * expand
        self.in_proj = nn.Linear(dim, dim_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=dim_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(
            dim_inner, dt_rank + self.d_state * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=dim_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim_inner))
        self.out_proj = nn.Linear(dim_inner, dim, bias=bias)




    def forward(self, x: Tensor):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)


        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        x_and_res = rearrange(x_and_res, "b l x -> b x l")
        (x, res) = x_and_res.split(
            split_size=[self.dim_inner, self.dim_inner], dim=1
        )

        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(rearrange(y, "b dim l -> b l dim"))

        return output

    def ssm(self, x: Tensor):
        """ - Algorithm 2 in Section 3.2 in the Mamba paper [1]

        Args:
            x: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, d_in, l)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()
        
        x_dbl = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_dbl)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y

    def selective_scan(self, u, delta, A, B, C, D):
       
        (b, d_in, l) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (Δ, A, B)  (see Section 2 Equation 4 in the Mamba paper [1])
        # Note that B is parameterized directly
        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b d_in l n"))
        deltaB_u = einsum(
            delta, B, u, "b l d_in, b l n, b d_in l -> b d_in l n"
        )

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = torch.zeros((b, d_in, n)).cuda()
        ys = []
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i, :], "b d_in n , b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=2)  # (b d_in l)

        if D is not None:
            y = y + u * rearrange(D, "d_in -> d_in 1")
        return y


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dim_mult):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dim_mult = dim_mult

        # Calculate the intermediate dimension
        dim_inner = int(self.hidden_size * self.dim_mult)

        # Initialize layers
        self.gate_proj = nn.Linear(self.hidden_size, dim_inner, bias=False).cuda()

        self.up_proj = nn.Linear(self.hidden_size, dim_inner, bias=False).cuda()

        self.down_proj = nn.Linear(dim_inner, self.hidden_size, bias=False).cuda()


    def forward(self, x):
        gate_output = F.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_output = self.down_proj((gate_output * up_output))
        return down_output
    
class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        dim = int ,
        multi = int , 
    ):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(dim, multi).cuda() for i in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)

class MOEMamba(nn.Module):

    def __init__(
        self,
        dim: int,
        n_blocks: int,
        dropout: float,
        heads: int,
        d_state: int,
        experts: int,
        horizon: int,
        eps : int,
        positive : bool = True,
        expansion_rate: int = 2, # 2 ? 
        *args,
        **kwargs,
    ):
        super(MOEMamba, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.heads = heads
        self.n_blocks = n_blocks
        self.positive = positive
        if positive == True:
            self.sig = torch.nn.Sigmoid()

        self.hidden_dim = dim * expansion_rate

        # Set up the blocks
        self.mamba = nn.ModuleList([MambaBlock(dim=dim, d_state=d_state, *args, **kwargs).cuda() for _ in range(self.n_blocks)])
        self.feed = nn.ModuleList([MoE(experts, num_experts_per_tok=2, dim=dim , multi=2).cuda() for _ in range(self.n_blocks)])


        self.linear = nn.Linear(dim, dim, bias=False)

        self.out = nn.Linear(dim, dim, bias=False)
        self.out.weight = self.linear.weight 

        self.head = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, horizon, bias=False),
        )

        #self.feed = FeedForward(dim,  4)
        self.rmsnormf = RMSNorm(dim  , eps=eps).cuda()
        self.rmsnorma = RMSNorm(dim , eps=eps).cuda()


    def forward(self, x):
        x  = x.unsqueeze(0).permute(0, 2, 1) # => batch=1, context, feature
        x = self.linear(x)

        for mamba, feed in zip(self.mamba, self.feed):
            resi = x 
            x = self.rmsnorma(x)
            x = mamba(x)
            x = x + resi
            resi = x 
            x = feed(x)
            x = self.rmsnormf(x)
            x = x + resi

        x = self.out(x)
        x = self.head(x)
        x  = x.squeeze(0).permute(1, 0)  # => feature ,predictions till the horizon
        if self.positive == True:
            x = self.sig(x)
        return x
    
    def init_weights(self):
        self.apply(init_weights)

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding, nn.LayerNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias') or pn in ('pos_emb', 'global_pos_emb'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight')  and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                if fpn.startswith('mamba'):
                    if pn in ['A_log', 'D']:
                        decay.add(fpn) 
        # validate that we considered every parameter
                    
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0,\
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict],
                "weight_decay": weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict],
                "weight_decay": 0.0
            },
        ]
        self.eps = eps

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer
    
def init_weights(model):
    if isinstance(model, nn.Linear):
        # Initialize linear layers with Xavier (Glorot) initialization
        init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0.01)

    elif isinstance(model, nn.Conv1d):
        # Initialize Conv2d layers with Kaiming Normal (He initialization)
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)

    elif isinstance(model, nn.Embedding):
        # Initialize embedding layers with Xavier (Glorot) initialization
        init.xavier_uniform_(model.weight)

def load_and_split_dataset(context_len, horizon, load=True):
    # Load the dataset from the CSV file or the Hugging Face repository
    if load:
        dataset = datasets.load_dataset('Ammok/apple_stock_price_from_1980-2021')
        dataset = dataset['train']
        dataset = dataset.remove_columns('Date')
        context_windows = dataset.select(range(context_len)).to_pandas()
        tar_windows = dataset.select(range(context_len, context_len + horizon)).to_pandas()
    else:
        dataset = pd.read_csv("apple/apple_stock_price.csv")
        context_windows = dataset.iloc[:context_len, 1:]
        tar_windows = dataset.iloc[context_len:context_len + horizon, 1:]

    # Convert the context windows and target windows to NumPy arrays
    context_windows = context_windows.to_numpy(dtype=np.float32)
    tar_windows = tar_windows.to_numpy(dtype=np.float32)

    context_windows = torch.from_numpy(context_windows).float().cuda()
    tar_windows = torch.from_numpy(tar_windows).float().cuda()

    return context_windows, tar_windows

context_len = 256
horizon = 32
experts = 4
positive = True   #all datapoints  are positive
n_blocks = 8
heads = 8

model = MOEMamba(
    dim = context_len,
    n_blocks = n_blocks,
    dropout = 0.1,
    heads = heads,
    d_state = 16,
    experts = experts,
    horizon=horizon,
    eps=0.001,
    positive=positive, #trues if only positive values.
).cuda()
model.init_weights() 

learning_rate = 0.000001
betas = (0.95, 0.999)

optimizer = model.configure_optimizers( weight_decay=0.1, learning_rate=learning_rate,  betas = betas, eps=0.001)
criterion = nn.CrossEntropyLoss()

context_window, tar = load_and_split_dataset(context_len, horizon)

# Create a list to store the loss values
losses = []
fig, ax = plt.subplots()

# Train the model
def train(model, criterion, optimizer, x_batch, y_batch, num_epochs=1000):
    # Set the model to training mode
    model.train()
    plt.suptitle('Training Loss')
        # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Zero the gradients of the optimizer
        optimizer.zero_grad()
        model.zero_grad()
        output_tensor = model(x_batch)
        loss = criterion(output_tensor, y_batch)
        loss.backward()
        # Update the weights
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(f"{name}: {param.grad.data.abs().mean()}")
        losses.append(loss.item())
        ## Update the data in the subplots
        ax.plot(losses, label='Training Loss')

        ## Call the draw method to update the plot
        plt.draw()
        plt.pause(0.01)

train(model, criterion, optimizer, context_window, tar)
