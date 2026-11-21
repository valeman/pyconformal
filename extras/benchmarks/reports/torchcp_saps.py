import torch
from importlib.metadata import version
from torchcp.classification.score import SAPS
print("torchcp", version("torchcp"))
probs = torch.tensor([[0.6, 0.3, 0.1]])
s = SAPS(score_type="identity", randomized=False)
print("all labels      :", s._calculate_all_label(probs.clone()))
print("true label = 0  :", s._calculate_single_label(probs.clone(), torch.tensor([0])))

# coverage: 10 classes, 5,000 calibration and 50,000 test points, target 90%
n_cal, n_test, K, alpha = 5000, 50000, 10, 0.1
for SCALE in (3.0, 6.0, 10.0):
    torch.manual_seed(0)
    logits = SCALE * torch.randn(n_cal + n_test, K)
    p = torch.softmax(logits, dim=1)
    y = torch.multinomial(p, 1).squeeze(1)
    for randomized in (True, False):
        s = SAPS(score_type="identity", randomized=randomized)
        cal = s(p[:n_cal], y[:n_cal])
        k = int(torch.ceil(torch.tensor((n_cal + 1) * (1 - alpha))).item())
        q = torch.sort(cal).values[k - 1]
        sets = s(p[n_cal:]) <= q
        cov = sets[torch.arange(n_test), y[n_cal:]].float().mean().item()
        print(f"logit scale {SCALE}, top-1 accuracy {(p.argmax(1) == y).float().mean().item():.2f}, randomized={randomized}: coverage {cov:.3f}, mean set size {sets.sum(1).float().mean().item():.2f}")
