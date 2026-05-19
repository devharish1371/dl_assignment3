import torch
from torch.optim.lr_scheduler import LRScheduler
import torch.optim as optim

class NoamScheduler(LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        curr_step = max(1, self.last_epoch + 1)
        return (self.d_model ** -0.5) * min(curr_step ** -0.5, curr_step * (self.warmup_steps ** -1.5))

    def get_lr(self) -> list:
        scale_factor = self._get_lr_scale()
        return [base_learning_rate * scale_factor for base_learning_rate in self.base_lrs]

def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list:
    temp_model = torch.nn.Linear(1, 1)
    opt = optim.Adam(temp_model.parameters(), lr=1.0)
    sched = NoamScheduler(opt, d_model=d_model, warmup_steps=warmup_steps)
    
    history_of_lr = []
    for _ in range(total_steps):
        history_of_lr.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return history_of_lr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    D_MOD = 512
    W_STEPS = 4000
    T_STEPS = 20_000
    
    learning_rates = get_lr_history(D_MOD, W_STEPS, T_STEPS)
    plt.figure(figsize=(9, 4))
    plt.plot(learning_rates)
    plt.axvline(W_STEPS, color="red", linestyle="--", label=f"warmup_steps={W_STEPS}")
    plt.xlabel("Training Step")
    plt.ylabel("LR")
    plt.title(f"Noam Scheduler (d_model={D_MOD})")
    plt.legend()
    plt.tight_layout()
    plt.show()