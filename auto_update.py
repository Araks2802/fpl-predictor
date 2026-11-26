"""Decide what to update, then do it. Run by GitHub Actions (.github/workflows/update.yml) every 10 minutes.
  1. WEEKLY RETRAIN   a gameweek has finished (FPL has added the bonus points) and we have not trained on it yet:
                      full fetch, retrain the points and chance models, new predictions. Takes about 6 minutes.
  2. CHECKED RETRAIN  FPL has since finished checking that gameweek's data (data_checked). Same as 1, so any late
                      corrections are picked up. Usually changes very little.
  3. LATE REFRESH     the next deadline is 2.5 hours away or less (runs about 2 hours and 1 hour before it):
                      latest injury news, chance of playing and prices, new predictions. No retraining.
  4. DAILY REFRESH    nothing above, and the last update is over 20 hours old: same as the late refresh.
  Otherwise nothing changes, and the workflow commits nothing.

Waiting for the final whistle: GitHub often starts scheduled runs late, or skips them, sometimes for hours. So when the last
match of a gameweek is on (it kicked off up to 4 hours ago, or kicks off within the next hour), a run does not just exit: it
stays and checks FPL every 5 minutes until the gameweek is marked finished, then retrains straight away. One such run is
enough, and it is capped at WATCH_MAX_H hours so it never hits the job's time limit.

  python auto_update.py           decide and run
  python auto_update.py --dry     only print what it would do
  FPL_FORCE=late python auto_update.py   force an action (late / daily / retrain), for testing
"""
import json, os, subprocess, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).parent; DATA = ROOT / 'data'; CUR = DATA / '2026-27'
LOG = DATA / 'update_log.json'
PY = sys.executable
ENV = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '2'))
ENV.pop('FINAL', None); ENV.pop('FEATS', None)        # never let a test setting leak into the real run
LATE_WINDOW_H, LATE_GAP_MIN, DAILY_H = 2.5, 50, 20
WATCH_BEFORE_H, WATCH_AFTER_H, WATCH_EVERY_MIN, WATCH_MAX_H = 1.0, 4.0, 5, 4.5


def run(*args):
    print('$', ' '.join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, env=ENV, check=True)
def now(): return datetime.now(timezone.utc)
def parse(t): return datetime.fromisoformat(t.replace('Z', '+00:00'))


def state():
    """Fresh gameweek status from FPL (a few seconds)."""
    run('fetch_current.py', '--light')
    boot = json.load(open(CUR / 'bootstrap.json')); fx = json.load(open(CUR / 'fixtures.json'))
    finished = [e['id'] for e in boot['events'] if e['finished']]
    checked = [e['id'] for e in boot['events'] if e['finished'] and e.get('data_checked')]
    cur = next((e for e in boot['events'] if e['is_current']), None)
    nxt = next((e for e in boot['events'] if e['is_next']), None)
    last_ko = None
    if cur and not cur['finished']:
        ks = [parse(f['kickoff_time']) for f in fx if f['event'] == cur['id'] and f.get('kickoff_time')]
        last_ko = max(ks) if ks else None
    return dict(last_done=max(finished, default=0), last_checked=max(checked, default=0), cur=cur, nxt=nxt, last_ko=last_ko)


log = json.load(open(LOG)) if LOG.exists() else {}
dry = '--dry' in sys.argv
backup = {f: (CUR / f).read_bytes() for f in ('bootstrap.json', 'fixtures.json') if (CUR / f).exists()}
S = state()
if 'last_full_gw' not in log:                          # first run: the models in the repo were trained up to the current state
    log['last_full_gw'] = S['last_done']
log.setdefault('checked_gw', log['last_full_gw'])

# ---- the last match of a gameweek is on: wait here for FPL to mark the gameweek finished, then retrain immediately
started = now()
def watching(S):
    if dry or os.environ.get('FPL_FORCE') or S['last_ko'] is None or S['last_done'] > log['last_full_gw']: return False
    since = (now() - S['last_ko']).total_seconds() / 3600
    return -WATCH_BEFORE_H <= since <= WATCH_AFTER_H and (now() - started).total_seconds() / 3600 < WATCH_MAX_H
if watching(S):
    print(f"GW{S['cur']['id']}: last match kicked off {S['last_ko']:%a %H:%M} UTC. Waiting for FPL to mark the gameweek finished...", flush=True)
    while watching(S):
        time.sleep(WATCH_EVERY_MIN * 60); S = state()
        print(f"  {now():%H:%M} UTC: finished GW {S['last_done']}", flush=True)

nxt = S['nxt']
hours_left = (parse(nxt['deadline_time']) - now()).total_seconds() / 3600 if nxt else None
mins_since = (now() - parse(log['updated_at'])).total_seconds() / 60 if log.get('updated_at') else 1e9
if S['last_done'] > log['last_full_gw']:
    action = 'weekly retrain'
elif S['last_checked'] > log['checked_gw']:
    action = 'checked retrain'
elif hours_left is not None and 0 < hours_left <= LATE_WINDOW_H and mins_since >= LATE_GAP_MIN:
    action = 'late team-news refresh'
elif mins_since >= DAILY_H * 60:
    action = 'daily refresh'
else:
    action = None
if os.environ.get('FPL_FORCE'):                       # testing: FPL_FORCE=late / daily / retrain
    action = {'late': 'late team-news refresh', 'daily': 'daily refresh', 'retrain': 'weekly retrain'}[os.environ['FPL_FORCE']]
hl = f'{hours_left:.1f} h' if hours_left is not None else 'none (season over)'
print(f"last finished GW {S['last_done']} (checked {S['last_checked']}) | trained on {log['last_full_gw']} (checked {log['checked_gw']}) | "
      f"next deadline in {hl} | last update {mins_since:.0f} min ago -> {action or 'nothing to do'}")
if dry or action is None:
    for f, b in backup.items(): (CUR / f).write_bytes(b)   # leave the files exactly as they were
    sys.exit(0)
run('fetch_history.py')          # past seasons are not in the repo, and the predictions need them too (form carries over seasons). Skips files already present.
if action in ('weekly retrain', 'checked retrain'):
    run('fetch_current.py'); run('train_mlp.py'); run('train_events.py')
    log['last_full_gw'] = S['last_done']
    if S['last_checked'] >= S['last_done']: log['checked_gw'] = S['last_done']
run('make_predictions.py')
log.update(updated_at=now().isoformat(timespec='minutes'), update_type=action, next_gw=nxt['id'] if nxt else None)
json.dump(log, open(LOG, 'w'), indent=1)
meta = json.load(open(DATA / 'meta.json'))
meta.update(updated_at=log['updated_at'], update_type=action, trained_gw=log['last_full_gw'])   # the app shows "updating" while trained_gw is behind
json.dump(meta, open(DATA / 'meta.json', 'w'))
print('done:', action)
