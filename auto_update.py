"""Decide what to update, then do it. Run every hour by GitHub Actions (.github/workflows/update.yml).

  1. WEEKLY RETRAIN   a gameweek has finished and FPL has checked its data, and we have not trained on it yet:
                      full fetch, retrain the points and chance models, new predictions.
  2. LATE REFRESH     the next deadline is 2.5 hours away or less (runs about 2 hours and 1 hour before it):
                      latest injury news, chance of playing and prices, new predictions. No retraining.
  3. DAILY REFRESH    nothing above, and the last update is over 20 hours old: same as the late refresh.
  Otherwise nothing changes, and the workflow commits nothing.

  python auto_update.py           decide and run
  python auto_update.py --dry     only print what it would do
  FPL_FORCE=late python auto_update.py   force an action (for testing)
"""
import json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent; DATA = ROOT / 'data'; CUR = DATA / '2026-27'
LOG = DATA / 'update_log.json'
PY = sys.executable
ENV = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '2'))
ENV.pop('FINAL', None); ENV.pop('FEATS', None)        # never let a test setting leak into the real run
LATE_WINDOW_H, LATE_GAP_MIN, DAILY_H = 2.5, 50, 20


def run(*args):
    print('$', ' '.join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, env=ENV, check=True)


def now(): return datetime.now(timezone.utc)
def parse(t): return datetime.fromisoformat(t.replace('Z', '+00:00'))


log = json.load(open(LOG)) if LOG.exists() else {}
dry = '--dry' in sys.argv
backup = {f: (CUR / f).read_bytes() for f in ('bootstrap.json', 'fixtures.json') if (CUR / f).exists()}
run('fetch_current.py', '--light')                     # fresh gameweek status, news, prices (seconds)
boot = json.load(open(CUR / 'bootstrap.json'))
done = [e['id'] for e in boot['events'] if e['finished'] and e.get('data_checked')]
last_done = max(done) if done else 0
nxt = next((e for e in boot['events'] if e['is_next']), None)
hours_left = (parse(nxt['deadline_time']) - now()).total_seconds() / 3600 if nxt else None
mins_since = (now() - parse(log['updated_at'])).total_seconds() / 60 if log.get('updated_at') else 1e9
if 'last_full_gw' not in log:                          # first run: the models in the repo were trained up to the current state
    log['last_full_gw'] = last_done

if last_done > log['last_full_gw']:
    action = 'weekly retrain'
elif hours_left is not None and 0 < hours_left <= LATE_WINDOW_H and mins_since >= LATE_GAP_MIN:
    action = 'late team-news refresh'
elif mins_since >= DAILY_H * 60:
    action = 'daily refresh'
else:
    action = None
if os.environ.get('FPL_FORCE'):                       # testing: FPL_FORCE=late / daily / retrain
    action = {'late': 'late team-news refresh', 'daily': 'daily refresh', 'retrain': 'weekly retrain'}[os.environ['FPL_FORCE']]
hl = f'{hours_left:.1f} h' if hours_left is not None else 'none (season over)'
print(f'last finished GW {last_done} | trained on {log["last_full_gw"]} | next deadline in {hl} | last update {mins_since:.0f} min ago -> {action or "nothing to do"}')

if dry or action is None:
    for f, b in backup.items(): (CUR / f).write_bytes(b)   # leave the files exactly as they were
    sys.exit(0)

if action == 'weekly retrain':
    run('fetch_current.py'); run('fetch_history.py'); run('train_mlp.py'); run('train_events.py')
    log['last_full_gw'] = last_done
run('make_predictions.py')
log.update(updated_at=now().isoformat(timespec='minutes'), update_type=action, next_gw=nxt['id'] if nxt else None)
json.dump(log, open(LOG, 'w'), indent=1)
meta = json.load(open(DATA / 'meta.json')); meta.update(updated_at=log['updated_at'], update_type=action); json.dump(meta, open(DATA / 'meta.json', 'w'))
print('done:', action)
