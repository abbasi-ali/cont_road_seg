import torch 
from torch.nn import functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_ewc_params(model, dl, bs, noise=None):
    eps_= []

    if noise != None:
        eps_ = noise

    norm_fact = len(dl)
    
    for b_ind, (x, y) in enumerate(dl):
        x, y = x.reshape(-1, 784).to(device), y.to(device)
        x = 2*x - 1

        if noise != None:
            noise_for_batch = eps_[b_ind*bs:min(eps_.shape[0], (b_ind+1)*bs)].to(device)
            x_tilt = torch.clamp(x + noise_for_batch, -1, 1)
        else:
            x_tilt = x 

        preds = model(x_tilt)
        loss = F.nll_loss(F.log_softmax(preds, dim=1), y)

        model.zero_grad()
        loss.backward()

        for p_ind, (n, p) in enumerate(model.named_parameters()):
            if hasattr(model, f"fisher_{n.replace('.', '_')}"):
                current_fisher = getattr(model, f"fisher_{n.replace('.', '_')}")
            else:
                current_fisher = 0

            new_fisher = current_fisher + p.grad.detach() ** 2 / norm_fact 
            model.register_buffer(f"fisher_{n.replace('.', '_')}", new_fisher)

    for p_ind, (n, p) in enumerate(model.named_parameters()):
        model.register_buffer(f"mean_{n.replace('.', '_')}", p.data.clone())

    model.zero_grad()


def compute_ewc_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        loss += (getattr(model, f"fisher_{n.replace('.', '_')}") * \
            (p - getattr(model, f"mean_{n.replace('.', '_')}")).pow(2)).sum()

    return loss / 2.

