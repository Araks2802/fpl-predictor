# FPL Points Predictor: how to explain it

## 1. The 30-second version

"I built a Fantasy Premier League assistant. You enter your FPL team ID and it predicts how many points each of your players will score in the coming gameweek, picks your best line-up and captain, and suggests transfers.

The predictions come from a PyTorch neural network blended with a gradient-boosting model, trained on about 122,000 player-match records from five seasons. A linear-programming optimiser then finds the best legal transfers under all the FPL rules. I tested everything on gameweeks the models had never seen: it ranks players clearly better than recent form, which is what most managers go on."

## 2. The 2-minute walkthrough (problem, data, model, decisions, results)

**The problem.** FPL managers decide every week who to start, who to captain and who to transfer. Most people go on recent form, which is noisy. I wanted a tool that gives an honest expected-points number for every player, then turns those numbers into decisions.

**The data.**
- The official FPL API (public, no key) for this season: every player's game-by-game stats, prices, injury news, fixtures and fixture difficulty, and any manager's team from their ID.
- A public GitHub archive (vaastav/Fantasy-Premier-League) for four past seasons, because the API only shows the current one.
- Joined into one table: one row per player per match, about 122,000 rows.

**The features (67 in total).** Only things known before kick-off:
- The player: points, minutes, xG, xA, bonus, saves and defensive actions over his last 3, 5, 10 and 20 games; per-90 rates; how often he starts; last season's level; price; position.
- The teams: goals and expected goals for and against for his team and the opponent, recent results, and a clean-sheet chance.
- Defensive contributions and bonus: how often he reached the threshold and got bonus recently, defensive actions and bonus-system points per 90.
- The match: home or away, his record against this opponent, his home/away record.
- The season: how many points each position is scoring right now, so the model adapts to rule changes.

**The model.**
- A PyTorch network: the features plus a learned position embedding, two hidden layers (128 and 64 units), ReLU, dropout, trained with AdamW and early stopping. I train five copies from different random starts and average them (an ensemble).
- A gradient-boosting model (scikit-learn) on the same features.
- The final prediction is the average of the two, which beat either one alone.
- Separate classifiers give the chance of a goal, an assist, a clean sheet, a defensive contribution, bonus and playing 60 minutes.

**Turning predictions into decisions.** A mixed-integer linear program (PuLP) chooses the squad, starting XI and captain for each of the next weeks to maximise expected points, subject to every FPL rule: 2/5/5/3 squad, max 3 per club, budget, valid formations, selling prices (you keep half of any price rise) and a 4-point hit per extra transfer. It solves in under a second.

**Results (tested on unseen gameweeks, regular starters).**

| Method | Ranking accuracy | Typical error | Avg actual points of each week's top 10 picks |
|---|---|---|---|
| Average of last 5 games | 0.175 | 3.34 | 4.6 |
| My model | 0.263 | 3.08 | 5.0 |

Ranking accuracy is the rank correlation between predicted and actual points within each gameweek. A second test on the whole of 2025-26 (38 gameweeks) gave 0.257, so the result holds up.

## 3. Why I made each decision (the "why" questions)

| Decision | Why |
|---|---|
| Split by time, never randomly | A random split lets the model learn from games played after the ones it is tested on. That is cheating and inflates results. |
| Compare against "last 5 games average" | A model is only worth something if it beats the simple thing people already do. |
| Neural net + gradient boosting blend | On tabular data gradient boosting is very strong; the net learns slightly different patterns. The blend beat both. |
| Ensemble of 5 nets | Each net makes different mistakes; averaging makes predictions steadier. |
| Early stopping, dropout, weight decay | FPL points are very noisy, so the model would otherwise memorise luck (overfitting). |
| Standardise features using training data only | Using test data to scale would leak information from the future. |
| Optimiser instead of "pick the top players" | Rules interact (budget, 3 per club, formations, hits). Only an exact search guarantees the best legal answer. |
| Did NOT add more weight to form | I tested it: 30% extra form made ranking worse (0.269 to 0.255), in two separate seasons. Hot streaks mostly fade. |
| Clean sheets modelled per team | A clean sheet is a team event, so team chance x his chance of playing 60 minutes is more sensible than a per-player guess. |
| App does no training | Predictions are precomputed, so the online app is fast and doesn't need PyTorch. |

## 4. Problems I hit and how I fixed them (recruiters love these)

1. **A data leak that made the model look great.** The first run scored 0.51 ranking accuracy. A sanity check showed Liverpool playing the same opponent six times in one week. Fixture IDs restart every season, so a join on fixture ID alone copied rows across seasons: 312,000 rows instead of 122,000. After fixing the join key, the honest score was about 0.25. *Lesson: when a result looks too good, look for a leak, and check row counts after every join.*
2. **Players on the wrong team.** Past-season files list a player's end-of-season club, so players who moved mid-season were linked to the wrong team for earlier games. I took each row's team from the fixture itself.
3. **Long-range projections drifting to average.** Future weeks have no results, so rolling averages filled with blanks. I held each player's form at today's level for all future weeks.
4. **Rule changes.** 2025-26 added defensive-contribution points and 2026-27 changed the bonus system, so older seasons score differently. I added a "current scoring level by position" feature so the model adapts.
5. **Engineering issues.** A macOS crash when PyTorch and scikit-learn load different threading libraries (fixed with environment settings), and a new PuLP version that ships without a solver (pinned a working version). The optimiser now also checks that a legal solution exists instead of returning nonsense.

## 5. Questions you might get, with answers

**"How accurate is it?"** "It ranks regular starters with a rank correlation of about 0.26 per gameweek, against 0.18 for recent form. Exact scores are hard: the typical error is about 3 points, because one goal changes a score by 4 to 10 points. The goal is better decisions, not perfect scores."

**"Why not just use FPL's own expected points?"** "This early in the season FPL's number is just recent form. My tests show form on its own is the weakest predictor I tried."

**"Is the model biased?"** "I checked calibration. Overall it predicts 1.36 points per player-match against an actual 1.37. This season it slightly under-predicts goalkeepers and forwards, probably because of the new bonus rules, and over a full previous season it slightly over-predicted, so there's no lasting bias."

**"Why PyTorch for tabular data?"** "Honestly, gradient boosting is about as good here. I used PyTorch to learn it and because embeddings (like the position embedding) extend naturally to player embeddings and sequence models. The blend of both was best."

**"What does the optimiser actually do?"** "It has a yes/no variable for every player being in the squad, in the XI each week, and captain each week. It maximises weighted expected points minus hit costs, with the FPL rules as constraints. It's exact, not a heuristic."

**"What would you do next?"** "Predict a distribution of points instead of a single number (useful for captaincy risk), player embeddings, a sequence model over each player's recent games, and an automated refresh before each deadline with late injury news."

**"Can it make transfers for me?"** "No, on purpose. FPL has no official way to authorise an app, so it would mean storing users' passwords and breaking FPL's terms. Instead it gives a ready-to-copy checklist."

## 6. Numbers to remember

- 5 seasons, about 122,000 player-match rows, 67 features
- PyTorch: 5-net ensemble, 2 hidden layers (128, 64), position embedding of 8
- Ranking accuracy 0.263 vs 0.175 for recent form; top-10 picks 5.0 vs 4.6 points
- Leak bug: 312,000 rows instead of 122,000; fake 0.51 became an honest 0.25
- Optimiser solves in under 1 second

## 7. Be honest about how you built it

If asked, say you built it with an AI coding assistant, and that you drove the decisions: what to predict, what to test, rejecting extra form weight after seeing the evidence, spotting that predictions looked too low, and asking for team form, matchup and event probabilities. Recruiters will probe the details, so before interviews work through the study notes (`notes/PyTorch_and_FPL_Notes.pdf`) and do exercises 1 to 4. Being able to change the network size and explain what happened is worth more than memorising this page.
