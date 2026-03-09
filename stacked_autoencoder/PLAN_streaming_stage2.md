# Implementation Plan: Streaming Batched GPU Evaluation for Stage-2 Feature Importance

## Problem

In `feature_selection.py`, the function `_compute_masking_importance()` (lines 258–361)
causes a CUDA OOM on a 12 GB GPU when run on the SWaT dataset (~500k rows, 51 features).

**Root cause:** Three memory anti-patterns in the importance-scoring loop:

1. **Line 296–299** — All sliding windows are eagerly materialized into one dense
   `np.array(sequences)` of shape `(N, seq_len, n_features)`.
2. **Line 302–304** — `train_test_split` copies the full array, and both halves are
   converted to `torch.Tensor`, doubling RAM usage.
3. **Lines 331–334 & 344–351** — The entire validation tensor is moved to GPU for a
   single forward pass. This happens once for the baseline and then again
   `n_features × n_repeats` times for corrupted copies. Each call allocates:
   - the input tensor on GPU
   - LSTM hidden states and the `h.unsqueeze(1).repeat(...)` expansion in the decoder
   - the full reconstruction output tensor

   Peak VRAM ≈ `val_tensor + recon_output + LSTM intermediates` which far exceeds 12 GB.

**Current workaround:** `main_swat.py` line 102 forces `fs_device = torch.device("cpu")`.
This avoids the OOM but makes Stage-2 significantly slower.

---

## Goal

Refactor `_compute_masking_importance()` so that **only one small batch** lives on GPU
at a time, reducing peak VRAM from "entire validation set" to approximately
`eval_batch_size × seq_len × n_features` (a few MB). This allows Stage-2 to run on GPU
again, and the CPU workaround in `main_swat.py` can be removed.

---

## File-by-file Changes

### 1. `feature_selection.py`

#### 1a. Add a lazy `_WindowDataset` class (new, ~15 lines)

Add a small `torch.utils.data.Dataset` subclass near the top of the file (after the
existing `_SmallLSTMAE` class, around line 48):

```python
class _WindowDataset(torch.utils.data.Dataset):
    """Lazy sliding-window dataset that avoids materializing all windows."""

    def __init__(self, data: np.ndarray, seq_len: int):
        # data: (timesteps, n_features), kept as float32 numpy
        self.data = data.astype(np.float32, copy=False)
        self.seq_len = seq_len
        self.n = data.shape[0] - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx : idx + self.seq_len])
```

- Windows are sliced on-the-fly; only one window is ever materialized per `__getitem__`.
- `np.float32` avoids repeated dtype conversion.

#### 1b. Add import for `Subset` (line 18)

Change the existing import:
```python
from torch.utils.data import DataLoader, TensorDataset
```
to:
```python
from torch.utils.data import DataLoader, TensorDataset, Subset
```

#### 1c. Add a `_batched_mse()` helper function (new, ~25 lines)

Add a helper that computes global MSE over a DataLoader using running accumulators.
Place it right before `_compute_masking_importance`:

```python
@torch.inference_mode()
def _batched_mse(model, loader, device, corrupt_fn=None):
    """Compute global MSE over a DataLoader, optionally corrupting each batch.

    Args:
        model: the trained _SmallLSTMAE in eval mode.
        loader: DataLoader yielding batches of shape (B, seq_len, n_features).
        device: torch device for inference.
        corrupt_fn: optional callable(batch_np) -> batch_np that corrupts
                    a numpy copy of the batch before inference.
                    If None, compute clean baseline MSE.

    Returns:
        Global MSE (float) = total_sse / total_elements.
    """
    total_sse = 0.0
    total_n = 0

    for batch in loader:
        clean = batch                           # CPU tensor from DataLoader
        if corrupt_fn is not None:
            corrupted_np = batch.numpy().copy()  # (B, seq_len, n_features)
            corrupted_np = corrupt_fn(corrupted_np)
            inp = torch.from_numpy(corrupted_np).to(device, non_blocking=True)
        else:
            inp = clean.to(device, non_blocking=True)

        target = clean.to(device, non_blocking=True)
        recon = model(inp)
        total_sse += torch.nn.functional.mse_loss(recon, target, reduction="sum").item()
        total_n += target.numel()

    return total_sse / total_n
```

**Important detail:** Use `reduction="sum"` and divide by total element count at the
end, NOT per-batch mean averaging (which would be biased when the last batch is smaller).

#### 1d. Rewrite `_compute_masking_importance()` body (lines 284–361)

Replace the current body with streaming logic. The function signature stays **identical**.

**New body outline:**

```python
def _compute_masking_importance(
    train_data, representative_indices, sequence_length, device,
    hidden_dim=64, num_epochs=15, batch_size=64, n_repeats=3,
):
    # --- Subset to representative features + scaling (unchanged) ---
    data = train_data[:, representative_indices].astype(np.float32)
    n_features = data.shape[1]
    medians = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqrs = q75 - q25
    iqrs[iqrs == 0] = 1.0
    data = (data - medians) / iqrs

    # --- Lazy windowed dataset + index-based train/val split ---
    full_dataset = _WindowDataset(data, sequence_length)
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size, shuffle=True,
    )
    eval_batch_size = max(batch_size, 512)   # can be larger since no gradients
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=eval_batch_size, shuffle=False,
    )

    # --- Build and train small AE (unchanged logic) ---
    model = _SmallLSTMAE(n_features, hidden_dim, sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"\nStage 2 — Training small LSTM-AE ({n_features} features, {num_epochs} epochs)...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  loss={epoch_loss / len(train_loader):.6f}")

    # --- Baseline MSE (batched) ---
    model.eval()
    baseline_mse = _batched_mse(model, val_loader, device)
    print(f"  Baseline val MSE: {baseline_mse:.6f}")

    # --- Per-feature importance via block permutation (batched) ---
    importance_scores = np.zeros(n_features)
    rng = np.random.RandomState(42)

    for fi in range(n_features):
        delta_sum = 0.0
        block_size = max(1, sequence_length // 4)
        for _ in range(n_repeats):
            def corrupt_fn(batch_np, _fi=fi, _bs=block_size, _rng=rng):
                for si in range(batch_np.shape[0]):
                    batch_np[si, :, _fi] = _block_permute(
                        batch_np[si, :, _fi], block_size=_bs, rng=_rng
                    )
                return batch_np

            mse_corrupted = _batched_mse(model, val_loader, device, corrupt_fn=corrupt_fn)
            delta_sum += mse_corrupted - baseline_mse
        importance_scores[fi] = delta_sum / n_repeats

    # --- Logging (unchanged) ---
    print("\n  Feature masking importance (ΔMSE):")
    ranked = np.argsort(importance_scores)[::-1]
    for rank, fi in enumerate(ranked):
        orig_idx = representative_indices[fi]
        print(f"    Feature {orig_idx}: {importance_scores[fi]:.6f}")

    return importance_scores
```

**Key differences from the original:**

| Aspect | Before (OOM) | After (streaming) |
|---|---|---|
| Window creation | Eager `np.array(sequences)` — all windows in RAM | Lazy `_WindowDataset.__getitem__` — one window at a time |
| Train/val split | `train_test_split(sequences)` copies entire array | `train_test_split(indices)` — splits integer indices only |
| Training DataLoader | `TensorDataset(train_tensor)` — full tensor in RAM | `Subset(full_dataset, train_idx)` — lazy |
| Baseline MSE | `model(val_tensor.to(device))` — entire val set on GPU | `_batched_mse(model, val_loader, device)` — one batch at a time |
| Corrupted MSE | `val_seq.copy()` + full tensor `.to(device)` per repeat | `corrupt_fn` applied per batch inside `_batched_mse` |
| MSE computation | `torch.mean(...)` on full tensors | `reduction="sum"` accumulated, divided by total count |
| Peak GPU VRAM | ~entire val set × 2–3 (input + recon + intermediates) | ~1 batch × 2–3 |

### 2. `main_swat.py`

#### 2a. Restore GPU device for feature selection (line 102)

Change:
```python
fs_device = torch.device("cpu")
```
to:
```python
fs_device = DEVICE
```

This removes the CPU workaround so Stage-2 runs on GPU again.

---

## What NOT to Change

- **`_SmallLSTMAE` class** — no changes needed.
- **`_block_permute()` function** — no changes needed.
- **`perform_feature_selection()` public API** — signature and logic unchanged.
- **Stages 0, 1, 3** — untouched.
- **`config.py`** — no changes needed.
- **Any other file** — no changes needed.

---

## Implementation Order

1. Add the `_WindowDataset` class to `feature_selection.py`.
2. Add `Subset` to the import line.
3. Add the `_batched_mse()` helper function.
4. Rewrite the body of `_compute_masking_importance()`.
5. Change `main_swat.py` line 102 to use `DEVICE` instead of `torch.device("cpu")`.

---

## Verification Criteria

### V1. No import or syntax errors

```bash
cd stacked_autoencoder
python -c "import feature_selection; print('OK')"
```
Expected: prints `OK` with no errors.

### V2. Feature selection runs to completion on GPU

```bash
cd stacked_autoencoder
python -c "
import numpy as np, torch
from feature_selection import perform_feature_selection

# Synthetic data: 5000 rows, 20 features (small enough for any GPU)
np.random.seed(42)
data = np.random.randn(5000, 20).astype(np.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
groups, dropped = perform_feature_selection(data, 20, 30, device)
print('Groups:', groups)
print('Dropped:', dropped)
print('PASS')
"
```
Expected: completes without OOM, prints encoder groups and `PASS`.

### V3. Importance scores are numerically consistent

The refactored code must produce **the same importance ranking** as the original
(up to minor floating-point differences from batched vs. full-tensor MSE).

To verify, temporarily run both old and new code on a small dataset and compare:
- The **rank order** of features by importance should be identical.
- The **absolute ΔMSE values** should match to within ~1e-5 relative tolerance.

### V4. Training loop still works

The training portion of `_compute_masking_importance` uses `Subset(full_dataset, train_idx)`
which yields individual tensors (not tuples). Verify the training loop correctly handles
this: `for batch in train_loader:` should yield `(B, seq_len, n_features)` tensors directly.

Note: `_WindowDataset.__getitem__` returns a tensor, so `DataLoader` will collate them
into a single tensor (not a tuple). The old code used `TensorDataset` which wraps in a
tuple, hence the old loop was `for (batch,) in train_loader:`. The new loop should be
`for batch in train_loader:`. Ensure this change is applied consistently.

### V5. Full SWaT pipeline runs to completion

```bash
cd stacked_autoencoder
python main_swat.py
```
Expected: Runs Stages 0–3 on GPU without OOM, then trains the final model and prints
evaluation results. This is the ultimate end-to-end verification.

### V6. No CPU workaround remains

Verify `main_swat.py` no longer contains `torch.device("cpu")` for feature selection.
The `fs_device` variable should use `DEVICE` (which is `cuda` when a GPU is available).

### V7. Memory footprint check (optional, manual)

If `nvidia-smi` is available, monitor peak GPU memory during Stage-2:
```bash
watch -n 0.5 nvidia-smi
```
Peak usage during Stage-2 should stay well under 12 GB (expected: < 2 GB for the
small LSTM-AE with batch_size=512).
