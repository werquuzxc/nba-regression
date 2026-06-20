# nba_regression.py

A multiple linear regression model built from scratch with NumPy (no scikit-learn) that predicts Anthony Edwards' points per game for the 2026/27 NBA season.

## How It Works

The model fits `points = w1 * season + w2 * usage_rate + b` using batch gradient descent:

1. Features are mean-centered (`x - x.mean()`) so gradient descent converges cleanly without feature scaling issues.
2. Predictions are computed each epoch as `X @ w + b`.
3. Loss is Mean Squared Error: `mean((pred - actual)^2)`.
4. Gradients (`dw`, `db`) are computed analytically and weights are updated via `w -= lr * dw`, `b -= lr * db`.
5. This repeats for 1000 epochs at a learning rate of 0.01.

## Features

| Feature | Description |
|---|---|
| `season` | Season number in Edwards' career (1–5, 2020/21 to 2024/25) |
| `usage_rate` | Usage rate (%), the share of team possessions he uses while on court |

Target: points per game.

## Key Insight

| Metric | Value |
|---|---|
| Usage rate weight | 0.87 (dominant) |
| Season weight | 0.02 (negligible) |
| MSE (usage rate only) | 0.326 |
| MSE (season + usage rate) | 0.044 |
| 2026/27 prediction | 28.6 pts/game |

Scoring growth tracks usage rate, not experience. The model learns almost nothing from season number alone — give Edwards more possessions and he scores more; time on its own isn't the driver.

## Requirements

- Python 3.8+
- NumPy

```bash
pip install numpy
```

## Usage

```bash
python nba_regression.py
```

Prints learned weights, bias, final MSE, and the 2026/27 prediction (assumes usage rate rises to 34.5%).
