simple = Toy implementation of MAMBA MOE.



more = i palyed around adding learned parameter XD to the attention(q*k) *v 

residual = x

learned paremeters XD

x' = Attention(x)
y = einsum (x', XD)
x = y * F.silu(XD)

x = x + residual
