import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import torch.nn.init as init
import matplotlib.pyplot as plt
import datasets

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, n_kv_heads, n_rep, head_dim)
            .reshape(bs, n_kv_heads * n_rep, head_dim)
        )

    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).cuda()
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(device=freqs.device)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    freqs_cis = freqs_cis[:x.shape[1], :]
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def rotate_half(x):
    return torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb4(q, k,k2,k3,k4, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    k_embed2 = (k2 * cos) + (rotate_half(k) * sin)
    k_embed3 = (k3 * cos) + (rotate_half(k) * sin)
    k_embed4 = (k4 * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed, k_embed2, k_embed3, k_embed4


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states_d = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
    

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super(RotaryEmbedding, self).__init__()
        
        self.dim = dim
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Initialize cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device, 
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bsz, n_heads, T, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class LlamaDynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
    Rotary Embedding extended with Dynamic NTK scaling.
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        super().__init__(dim, max_position_embeddings, base, device)


    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        
        # Recalculate base if seq_len exceeds max_position_embeddings
        if seq_len > self.max_position_embeddings:
            adjusted_base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    

    @staticmethod
    def print_grad(module, grad_input, grad_output):
        print(f"Gradients in FeedForward ({module}):")
        print("Gradient at output of the module:", grad_output)
        print("Gradient at input of the module:", grad_input)

class attention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p=None, attention_bias=False, rope_theta=10000):
        super().__init__()
        self.hidden_size = h_dim
        self.num_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.num_key_value_heads = n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = (self.head_dim) ** -0.5

        # Initialize the linear layers
        self.q_proj = nn.Linear(h_dim, n_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(h_dim, n_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(h_dim, n_heads * self.head_dim, bias=attention_bias)

        self.o_proj = nn.Linear(n_heads * self.head_dim, h_dim, bias=attention_bias)
        
        # Print the shapes of the weights
        #print("q_proj weight shape:", self.q_proj.weight.shape)
        #print("k_proj weight shape:", self.k_proj.weight.shape)
        #print("v_proj weight shape:", self.v_proj.weight.shape)
        #print("o_proj weight shape:", self.o_proj.weight.shape)
        # Rotary Embeddings Initialization
        self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(self.head_dim)
        self.position_ids = torch.arange(0, max_T).unsqueeze(0)


    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, output_attentions=False):
        bsz, q_len, _ = hidden_states.size()

        # Use normal weights for q, k, v projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # The rest of the attention mechanism as before
        q = self._shape(q, q_len, bsz)
        k = self._shape(k, q_len, bsz)
        v = self._shape(v, q_len, bsz)

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.position_ids)

        attn_weights = torch.einsum('bhsd,bhtd->bhst', q, k) / self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhst,bhtd->bhsd', attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output



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

class Test(nn.Module):

    def __init__(
        self,
        dim: int,
        max_t: int,
        dropout: float,
        heads: int,
        horizon: int,
        eps : int,
        positive : bool = True,
        expansion_rate: int = 2, # 2 ? 
    ):
        super(Test, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.heads = heads
        self.positive = positive
        if positive == True:
            self.sig = torch.nn.Sigmoid()

        self.hidden_dim = dim * expansion_rate
        xd =torch.ones(1, max_t, dim)
        self.XD =  nn.Parameter(torch.log(xd))

        # Set up the blocks
        self.attention = attention(dim, max_t, heads)
        
        #self.feed = nn.ModuleList([MoE(experts, num_experts_per_tok=2, dim=dim , multi=2).cuda() for _ in range(self.n_blocks)])
        self.feed = FeedForward(dim,  4)

        self.linear = nn.Linear(dim, dim, bias=False)

        self.out = nn.Linear(dim, dim, bias=False)
        self.out.weight = self.linear.weight 

        self.head = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, horizon, bias=False),
        )

        self.rmsnormf = RMSNorm(dim  , eps=eps).cuda()
        self.rmsnorma = RMSNorm(dim , eps=eps).cuda()


    def forward(self, x):
        x  = x.unsqueeze(0).permute(0, 2, 1) # => batch=1, context, feature
        x = self.linear(x)
    
        resi = x 
        x = self.rmsnorma(x)
        x = self.attention(x)
        xd = self.XD.float()
        fd = -torch.exp(xd)
        y = torch.einsum('bht,bht->bht', x, fd)
        x = y * F.silu(xd)
        x = x + resi

        resi = x 
        x = self.feed(x)
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
                if fpn.startswith('FD' or 'XD'):
                        decay.add(fpn) 
                if fpn.startswith('XD'):
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
positive = False   #all datapoints  are positive
heads = 8

model = Test(
    dim = context_len,
    max_t=6,
    dropout = 0.1,
    heads = heads,
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
