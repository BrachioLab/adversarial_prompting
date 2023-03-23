import math
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
try:
    from botorch.generation import MaxPosteriorSampling 
except:
    from .bo_torch_sampling import MaxPosteriorSampling 
from typing import Any
# based on TuRBO State from BoTorch

@dataclass
class TrustRegionState:
    dim: int
    best_value: float = -float("inf") 
    length: float = 1.0 # 0.8 
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32 
    success_counter: int = 0
    success_tolerance: int = 10 
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim ) / self.batch_size])
    #     )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1 # len(Y_next) # 1

    if state.success_counter >= state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    
    return state 


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points 
    Y,  # Evaluated scores
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling 
    num_restarts=10,
    raw_samples=256,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device('cuda'),
    absolute_bounds=None, 
):
    assert acqf in ("ts", "ei")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()  
    lb, ub = absolute_bounds 

    weights = torch.ones_like(x_center)
    if (lb is not None) and (ub is not None):
        weights = weights * (ub - lb)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, lb, ub) 
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, lb, ub) 
    else:
        weights = weights * 8 
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0 

    if acqf == "ei":
        try:
            ei = qExpectedImprovement(model.cuda(), Y.max().cuda() ) 
            X_next, _ = optimize_acqf(ei,bounds=torch.stack([tr_lb, tr_ub]).cuda(), q=batch_size, num_restarts=num_restarts,raw_samples=raw_samples,)
        except: 
            acqf = 'ts'

    if acqf == "ts":
        dim = X.shape[-1]
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        sobol = SobolEngine(dim, scramble=True) 
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        # Create a perturbation mask 
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.cuda()
        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.cuda()
        X_cand[mask] = pert[mask]
        # Sample on the candidate points 
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False ) 
        X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size )

    return X_next
