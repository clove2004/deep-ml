from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    best_loss = 1e10
    best_epoch, stop_epoch = -1, len(val_losses) - 1
    c_p = -1
    n = len(val_losses)
    for i in range (n):
        if best_loss - val_losses[i] > min_delta:
            best_loss = val_losses[i]
            best_epoch = i 
            c_p = 0
        else:
            c_p += 1
            if c_p == patience:
                stop_epoch = i
                break
    return stop_epoch, best_epoch