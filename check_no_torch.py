"""The online app must work without PyTorch or scikit-learn (they are only in requirements-train.txt). Block them and run every tab."""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('torch', 'sklearn', 'joblib'): raise ImportError(f'blocked {name}')
sys.meta_path.insert(0, Block())
from streamlit.testing.v1 import AppTest
fails = 0
for page in ['My team', 'Transfers', 'Planner', 'Captain', 'Differentials', 'Compare', 'Players', 'Fixtures', 'How it works']:
    at = AppTest.from_file('app.py', default_timeout=240); at.session_state['nav'] = page; at.session_state['eid'] = '1'; at.run()
    if at.exception: fails += 1; print(page, at.exception[0].value)
print('without torch / sklearn: failures', fails, '| torch loaded:', 'torch' in sys.modules)
