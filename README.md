# FPL Points Predictor

Enter your FPL team ID (or pick 15 players) to see expected points for the next three gameweeks, your best XI and captain, and the transfers that add the most.

- **Model:** a PyTorch neural network (ensemble of 5) blended with gradient boosting, trained on every player's game-by-game history from 2022-23 to now.
- **Transfers:** a linear-programming optimiser (PuLP) that follows the FPL rules: squad shape, 3 per club, budget, selling prices, free transfers, hits, Wildcard and Free Hit.
- **Study notes:** `notes/PyTorch_and_FPL_Notes.pdf`.

## Automatic updates

A GitHub Action (`.github/workflows/update.yml`) runs `auto_update.py` every hour:

- **After each gameweek** (once FPL has checked the data): full download, retrain both models, new predictions.
- **About 2 hours and 1 hour before each deadline**: late team-news refresh (injury flags, chance of playing, prices), new predictions.
- **Once a day otherwise**: the same light refresh.

It commits the new predictions, and Streamlit Cloud redeploys the app. Run it by hand from the Actions tab with "Run workflow".

## Manual refresh

```
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
python fetch_current.py      # latest results from the FPL API
python train_mlp.py          # retrain and re-test
python make_predictions.py   # predictions for the next 3 gameweeks
streamlit run app.py
```

The app itself only reads `data/predictions.parquet`, `data/meta.json` and `data/2026-27/bootstrap.json`, so the deployed version does not need PyTorch.

Data: the official Fantasy Premier League API and the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) archive.
