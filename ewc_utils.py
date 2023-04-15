import torch 
from torch.nn import functional as F 
from utils import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_ewc_params(model, dl, loss_type, t_cnt=None, multihead=False):

    norm_fact = len(dl)
    
    for b_ind, (x, y) in enumerate(dl):
        x, y = x.to(device), y.squeeze().to(device)

        preds = model(x)
        if multihead:
            preds = preds[t_cnt]

        if loss_type == "CE":
            loss = F.nll_loss(F.log_softmax(preds, dim=1), y)
        elif loss_type == "focal":
            criterion = FocalLoss(num_classes=34, gamma=3, alpha=1).to(device)
            loss = criterion(preds, y)

        model.zero_grad()
        loss.backward()

        for p_ind, (n, p) in enumerate(model.named_parameters()):
            if 'heads' not in n:
                if hasattr(model, f"fisher_{n.replace('.', '_')}"):
                    current_fisher = getattr(model, f"fisher_{n.replace('.', '_')}")
                else:
                    current_fisher = 0

                new_fisher = current_fisher + p.grad.detach() ** 2 / norm_fact 
                model.register_buffer(f"fisher_{n.replace('.', '_')}", new_fisher)

    for p_ind, (n, p) in enumerate(model.named_parameters()):
        if 'heads' not in n:
            model.register_buffer(f"mean_{n.replace('.', '_')}", p.data.clone())

    model.zero_grad()


def compute_ewc_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        if 'heads' not in n:
            loss += (getattr(model, f"fisher_{n.replace('.', '_')}") * \
                (p - getattr(model, f"mean_{n.replace('.', '_')}")).pow(2)).sum()

    return loss / 2.

