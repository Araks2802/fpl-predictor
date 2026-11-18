"""Build notes/PyTorch_and_FPL_Notes.pdf: the FPL rules, PyTorch from scratch, and how this project uses it."""
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'notes' / 'PyTorch_and_FPL_Notes.pdf'
ss = getSampleStyleSheet()
NAVY = colors.HexColor('#1f3a5f'); GREY = colors.HexColor('#555555'); LIGHT = colors.HexColor('#f3f5f8')
S = {
    'title': ParagraphStyle('t', parent=ss['Title'], fontSize=24, leading=29, textColor=NAVY, spaceAfter=6),
    'sub': ParagraphStyle('s', parent=ss['Normal'], fontSize=12, leading=16, textColor=GREY, spaceAfter=14),
    'h1': ParagraphStyle('h1', parent=ss['Heading1'], fontSize=17, leading=21, textColor=NAVY, spaceBefore=6, spaceAfter=8),
    'h2': ParagraphStyle('h2', parent=ss['Heading2'], fontSize=13, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=4),
    'p': ParagraphStyle('p', parent=ss['Normal'], fontSize=10.3, leading=14.6, spaceAfter=6),
    'b': ParagraphStyle('b', parent=ss['Normal'], fontSize=10.3, leading=14.6, leftIndent=14, bulletIndent=3, spaceAfter=3),
    'code': ParagraphStyle('c', fontName='Courier', fontSize=8.6, leading=11.2, backColor=LIGHT, borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10),
    'box': ParagraphStyle('x', parent=ss['Normal'], fontSize=10.3, leading=14.6, backColor=colors.HexColor('#e8f1fb'), borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=12),
}
story = []
P = lambda t, s='p': story.append(Paragraph(t, S[s]))
H1 = lambda t: story.append(Paragraph(t, S['h1']))
H2 = lambda t: story.append(Paragraph(t, S['h2']))
def B(items):
    for t in items: story.append(Paragraph(t, S['b'], bulletText='\u2022'))
def C(code): story.append(Preformatted(code.strip('\n'), S['code']))
def BOX(t): story.append(Paragraph(t, S['box']))
def T(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTSIZE', (0, 0), (-1, -1), 9),
                           ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]), ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.append(t); story.append(Spacer(1, 8))
def footer(c, d):
    c.saveState(); c.setFont('Helvetica', 8); c.setFillColor(GREY)
    c.drawString(2 * cm, 1.2 * cm, 'PyTorch and FPL: study notes'); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'Page {d.page}'); c.restoreState()

# ======================================================================= cover and contents
story += [Spacer(1, 30)]
P('PyTorch and FPL', 'title')
P('Study notes for the FPL Points Predictor project: the game rules, PyTorch from first principles, and a walk through the code we wrote.', 'sub')
BOX('<b>How to use these notes.</b> Read Part 1 once so the rules are clear. Part 2 teaches PyTorch with small examples you can type into Python. '
    'Part 3 connects each idea to a file in the project. Part 4 is about testing honestly, including a real bug we hit. Part 5 has exercises. '
    'Every code example runs with <font face="Courier">/opt/anaconda3/bin/python3</font>.')
P('<b>Contents</b>')
B(['Part 1. The FPL rules the tool follows', 'Part 2. PyTorch from scratch: tensors, gradients, models, training', 'Part 3. How this project works, file by file',
   'Part 4. Testing a model honestly (and the bug that fooled us)', 'Part 5. Exercises and next steps'])
story.append(PageBreak())

# ======================================================================= part 1: rules
H1('Part 1. The FPL rules (2026/27)')
P('These are the rules the optimiser has to respect. They come from the official 2026/27 rules (checked in November 2026).')
H2('Squad and team')
B(['<b>Squad:</b> 15 players. 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards.', '<b>Budget:</b> £100.0m to start. Your squad value then moves with prices.',
   '<b>Club limit:</b> at most 3 players from any one Premier League club.',
   '<b>Starting XI:</b> 1 goalkeeper, at least 3 defenders, at least 2 midfielders and at least 1 forward. The other 4 sit on the bench.',
   '<b>Captain:</b> scores double. If he plays no minutes, the vice-captain is doubled instead. If neither plays, nobody is doubled.',
   '<b>Automatic substitutes:</b> if a starter plays no minutes, the first bench player who played and keeps the formation legal comes on. The reserve goalkeeper can only replace the goalkeeper.'])
H2('Scoring')
T([['Action', 'GK', 'DEF', 'MID', 'FWD'], ['Up to 60 minutes', '1', '1', '1', '1'], ['60 minutes or more', '2', '2', '2', '2'], ['Goal', '10', '6', '5', '4'],
   ['Assist', '3', '3', '3', '3'], ['Clean sheet (60+ minutes)', '4', '4', '1', '0'], ['Every 3 saves', '1', '-', '-', '-'], ['Penalty save', '5', '-', '-', '-'],
   ['Defensive contribution', '-', '2 (10+ CBIT)', '2 (12+ CBIRT)', '2 (12+ CBIRT)'], ['Every 2 goals conceded', '-1', '-1', '-', '-'],
   ['Bonus (top 3 in BPS)', '3 / 2 / 1', '3 / 2 / 1', '3 / 2 / 1', '3 / 2 / 1'], ['Yellow / red card', '-1 / -3', '-1 / -3', '-1 / -3', '-1 / -3'],
   ['Own goal / missed penalty', '-2 / -2', '-2 / -2', '-2 / -2', '-2 / -2']], [5.2 * cm, 2.4 * cm, 2.6 * cm, 2.8 * cm, 2.8 * cm])
P('CBIT means clearances, blocks, interceptions and tackles. CBIRT adds recoveries. Defensive contribution points are capped at 2 per match. '
  'For 2026/27 the bonus system was changed (for example, one BPS point per three CBIs instead of two), and scores become final at 09:00 UK time the day after a gameweek ends.')
H2('Transfers and money')
B(['<b>Free transfers:</b> 1 per gameweek. Unused ones roll over, up to a maximum of 5.', '<b>Hits:</b> each transfer beyond your free ones costs 4 points.',
   '<b>Selling price:</b> if a player rose in price since you bought him, you get only half of the rise, rounded down to £0.1m. Bought at £7.5m, now £7.8m: you sell for £7.6m. Falls are passed on in full.',
   '<b>Price changes</b> happen overnight, driven by how many managers transfer a player in or out.'])
H2('Chips')
P('There are two sets of chips: one for Gameweeks 1 to 19 (must be used before the GW19 deadline) and one for Gameweeks 20 to 38. Only one chip per gameweek.')
B(['<b>Wildcard:</b> unlimited permanent transfers with no hits.', '<b>Free Hit:</b> unlimited transfers for one gameweek, then your old squad comes back. Not in GW1, and not in two weeks running.',
   '<b>Bench Boost:</b> your bench players score too.', '<b>Triple Captain:</b> your captain scores triple.',
   'Saved free transfers are kept when you play a Wildcard or Free Hit.'])
H2('Blank and double gameweeks')
P('In a <b>blank gameweek</b> some clubs have no match, so their players score nothing. In a <b>double gameweek</b> a club plays twice and its players can score in both. '
  'The tool adds up a player\'s expected points across all of his fixtures in a gameweek, so doubles and blanks are handled automatically.')
story.append(PageBreak())

# ======================================================================= part 2: pytorch
H1('Part 2. PyTorch from scratch')
P('PyTorch is a Python library for building and training neural networks. Four ideas carry almost everything: <b>tensors</b> (arrays), '
  '<b>autograd</b> (automatic gradients), <b>modules</b> (models made of layers) and the <b>training loop</b> (repeat: predict, measure the error, adjust).')

H2('2.1 Tensors')
P('A tensor is an array of numbers, like a NumPy array, that can also live on a GPU and remember how it was computed. A single number is a 0-D tensor, a list is 1-D, a table is 2-D.')
C('''
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # a 2 x 2 table
print(x.shape)          # torch.Size([2, 2])
print(x * 10)           # works element by element
print(x @ x)            # matrix multiplication
print(x.mean(dim=0))    # mean of each column: tensor([2., 3.])
y = torch.zeros(3, 5); z = torch.randn(3, 5)   # zeros, random normal numbers
a = x.numpy(); b = torch.from_numpy(a)        # to and from NumPy
''')
P('Two things to watch: the <b>dtype</b> (use <font face="Courier">float32</font> for model inputs, <font face="Courier">int64</font> for category numbers) '
  'and the <b>shape</b>. Most PyTorch errors are shape mismatches, so print shapes often.')

H2('2.2 Autograd: gradients for free')
P('Training means nudging numbers (the <b>weights</b>) to make the error smaller. To know which way to nudge, we need the <b>gradient</b>: how much the error changes when each weight changes. '
  'PyTorch records every operation on a tensor marked <font face="Courier">requires_grad=True</font>, then <font face="Courier">.backward()</font> works out all the gradients.')
C('''
w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5) ** 2          # error is smallest when w = 5
loss.backward()              # compute d(loss)/dw
print(w.grad)                # tensor(-4.)  -> increase w to reduce the loss
''')
P('The gradient is -4: increasing w lowers the loss. Gradient descent repeats a small step against the gradient:')
C('''
w = torch.tensor(3.0, requires_grad=True)
for step in range(50):
    loss = (w - 5) ** 2
    loss.backward()
    with torch.no_grad():        # the update itself should not be recorded
        w -= 0.1 * w.grad        # 0.1 is the learning rate
    w.grad.zero_()               # gradients add up unless you reset them
print(w)                         # close to 5
''')
BOX('That loop is all of deep learning in miniature. A real model just has millions of weights in place of one, and the loss compares predictions with real results.')

H2('2.3 A model: nn.Module and layers')
P('You build a model by subclassing <font face="Courier">nn.Module</font>. You list the layers in <font face="Courier">__init__</font> and say how data flows in <font face="Courier">forward</font>.')
B(['<b>nn.Linear(a, b)</b>: takes a numbers in and gives b numbers out, each a weighted sum plus a constant. These weights are what training learns.',
   '<b>nn.ReLU()</b>: replaces negative numbers with 0. Without a non-linear step like this, stacked Linear layers would collapse into one straight-line model.',
   '<b>nn.Dropout(p)</b>: during training, randomly zeroes a share p of the numbers, so the network cannot rely on any single path. It reduces overfitting.',
   '<b>nn.Embedding(n, d)</b>: a lookup table that gives each of n categories (for us, the 4 positions) its own learned vector of d numbers.'])
C('''
import torch.nn as nn
class TinyNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 1))
    def forward(self, x):
        return self.layers(x).squeeze(1)   # shape (batch,) not (batch, 1)

net = TinyNet(n_in=3)
print(net(torch.randn(5, 3)))      # 5 predictions from an untrained net
print(sum(p.numel() for p in net.parameters()))   # 81 weights to learn
''')

H2('2.4 Loss and optimiser')
P('The <b>loss</b> is a single number that says how wrong the predictions are. For points we use mean squared error (MSE): the average of (prediction minus actual) squared. '
  'The <b>optimiser</b> does the weight update for you. <font face="Courier">AdamW</font> is a good default: it adapts the step size for each weight, and its weight decay gently pulls weights towards zero.')
C('''
lossf = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
''')

H2('2.5 The training loop')
P('Training goes through the data in small <b>batches</b>, and one full pass over the data is an <b>epoch</b>. Each batch follows the same five steps:')
C('''
for epoch in range(20):
    net.train()                          # dropout ON
    perm = torch.randperm(len(X))        # shuffle the rows each epoch
    for i in range(0, len(X), 512):      # batches of 512 rows
        idx = perm[i:i + 512]
        opt.zero_grad()                  # 1. clear old gradients
        pred = net(X[idx])               # 2. forward pass
        loss = lossf(pred, y[idx])       # 3. how wrong?
        loss.backward()                  # 4. gradients
        opt.step()                       # 5. update weights
    net.eval()                           # dropout OFF for checking
    with torch.no_grad():
        val_loss = lossf(net(X_val), y_val).item()
    print(epoch, val_loss)
''')
P('<b>Why batches?</b> One gradient from 512 rows is a good enough estimate and much faster than using all 275,000 rows for every step. '
  '<b>train() and eval()</b> switch dropout on and off. <b>no_grad()</b> skips gradient bookkeeping when you are only predicting.')

H2('2.6 Overfitting and how to stop it')
P('A network can memorise the training rows, including their luck, and then predict new games badly. That is <b>overfitting</b>. We use four defences:')
B(['<b>A validation set:</b> rows the model does not train on, used to check progress after each epoch.',
   '<b>Early stopping:</b> keep the weights from the best validation epoch, and stop when it has not improved for 5 epochs.',
   '<b>Dropout and weight decay:</b> both make it harder to memorise.',
   '<b>An ensemble:</b> train 5 networks from different random starts and average them. Each makes different mistakes, and the average is steadier.'])

H2('2.7 Preparing the inputs')
P('Networks train badly when inputs are on very different scales (price around 6, minutes around 70). We <b>standardise</b> each feature: subtract its mean and divide by its standard deviation, '
  'using numbers from the <b>training rows only</b>. Missing values become 0, which after standardising means "average".')
C('''
mu, sd = train[F].mean(), train[F].std()
x = ((df[F] - mu) / sd).fillna(0).clip(-6, 6).values.astype('float32')
X = torch.tensor(x)
''')

H2('2.8 Devices, saving and loading')
C('''
device = 'mps' if torch.backends.mps.is_available() else 'cpu'   # Apple GPU on your Mac
net.to(device); X = X.to(device)          # model and data must be on the same device

torch.save(net.state_dict(), 'model.pt')  # save only the learned weights
net2 = TinyNet(3); net2.load_state_dict(torch.load('model.pt')); net2.eval()
''')
P('Our model is small, so the CPU is fast enough and that is what the project uses. The Apple GPU (MPS) helps with bigger models such as image or video networks.')
story.append(PageBreak())

# ======================================================================= part 3: project
H1('Part 3. How this project works')
T([['File', 'What it does'], ['fetch_current.py', 'Downloads this season from the official FPL API: every player\'s game history, fixtures and prices.'],
   ['fpl_data.py', 'Joins four past seasons and this one into one table, a row per player per fixture, and builds the input features.'],
   ['fpl_model.py', 'The PyTorch network (PointsNet), the feature scaler and a predict function.'],
   ['train_mlp.py', 'Trains 5 networks plus a gradient-boosting model, tests them on unseen gameweeks, then retrains on everything and saves.'],
   ['make_predictions.py', 'Writes expected points for the next 3 gameweeks to data/predictions.parquet.'],
   ['fpl_engine.py', 'Loads a manager\'s team, works out selling prices and free transfers, and runs the transfer optimiser.'],
   ['app.py', 'The Streamlit app.'], ['test_app.py', 'Clicks through every page of the app and reports any errors.']], [4.2 * cm, 12.0 * cm])

H2('3.1 The data')
P('Past seasons (2022-23 to 2025-26) come from the public <font face="Courier">vaastav/Fantasy-Premier-League</font> collection on GitHub. This season comes straight from the API. '
  'That gives about 122,000 rows, each one a player in a fixture with his points, minutes, goals, assists, expected goals (xG), expected assists (xA), bonus, saves and defensive actions.')
H2('3.2 Features: only what was known before kick-off')
P('Each feature is built from <b>earlier games only</b>. The code shifts every player\'s history down one row before averaging, so a match never sees its own result:')
C('''
h[f'{c}_r{w}'] = gcode[c].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
''')
B(['Recent averages over the last 3, 5 and 10 games: points, minutes, BPS, xG, xA, defensive contributions, saves and bonus.',
   'How often he started in his last 5 games, games seen so far, and his average points this season.',
   'Price, position, home or away, and the number of fixtures in the gameweek.',
   'Team strength: goals his team and the opponent scored and conceded over their last 8 games, pulled towards the league average when there are few games.',
   'An era flag for defensive contribution points, which began in 2025-26.'])
H2('3.3 The network')
C('''
class PointsNet(nn.Module):
    def __init__(self, n_feat, hidden=128, drop=0.2):
        super().__init__()
        self.pos = nn.Embedding(4, 8)                 # GK, DEF, MID, FWD
        self.body = nn.Sequential(
            nn.Linear(n_feat + 8, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden // 2, 1))
    def forward(self, x, pos):
        return self.body(torch.cat([x, self.pos(pos)], dim=1)).squeeze(1)
''')
P('The position goes in as an <b>embedding</b>, so the network learns its own way to describe each position. A defender\'s clean sheet and a forward\'s goals then '
  'shape the prediction differently. <font face="Courier">torch.cat</font> joins the 36 features and the 8 position numbers into one input row of 44.')
H2('3.4 Training')
P('<font face="Courier">fit_nets</font> in train_mlp.py is the loop from section 2.5 with early stopping, run 5 times with different seeds. The data is split <b>by time</b>, never at random: '
  'train on everything up to 2025-26 Gameweek 28, validate on Gameweeks 29 to 38, and test on this season from Gameweek 3. A random split would let the model learn from games played after the ones it is tested on.')
H2('3.5 From points to transfers')
P('The prediction for a fixture is the average of the PyTorch ensemble and a gradient-boosting model. It is then scaled by the player\'s chance of playing if he is flagged. '
  'The optimiser in fpl_engine.py uses <b>linear programming</b> (the PuLP library). It has a yes/no variable for every player in the squad, in the starting XI each week and as captain each week. '
  'It maximises expected points over the next 3 gameweeks (weights 1.0, 0.8 and 0.6), minus 4 points per hit, while keeping every FPL rule from Part 1. '
  'This is not machine learning. It is an exact search over every legal squad, and it takes under a second.')
H2('3.6 Running it each week')
C('''
cd /Users/pranav_arakkal/Documents/Portfolio/Website/FPL
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1   # avoids a macOS crash (see Part 4)
/opt/anaconda3/bin/python3 fetch_current.py
/opt/anaconda3/bin/python3 train_mlp.py
/opt/anaconda3/bin/python3 make_predictions.py
/opt/anaconda3/bin/streamlit run app.py
''')
story.append(PageBreak())

# ======================================================================= part 4: testing
H1('Part 4. Testing a model honestly')
P('A model is only as good as its score on games it has never seen. We test on 2026-27 from Gameweek 3, and compare against the simplest sensible guess: a player\'s average over his last 5 games.')
sc = pd.read_csv(ROOT / 'models' / 'test_scores.csv')
nm = {'last5': 'Average of last 5 games', 'gbm': 'Gradient boosting', 'mlp': 'PyTorch ensemble', 'blend': 'Blend (used in the app)'}
for grp, ttl in (('starters', 'Regular starters (45+ minutes a game over the last 3)'), ('all', 'All players')):
    H2(ttl)
    s = sc[sc.group == grp]
    T([['Method', 'Typical error (RMSE)', 'Average error (MAE)', 'Ranking accuracy']] +
      [[nm[r.predictor], f'{r.RMSE:.2f}', f'{r.MAE:.2f}', f'{r._4:.3f}'] for r in s.itertuples()], [5.6 * cm, 3.6 * cm, 3.6 * cm, 3.4 * cm])
P('<b>Ranking accuracy</b> is the rank correlation between predicted and actual points within each gameweek (1 would be a perfect ranking, 0 no better than chance). '
  'The neural network and gradient boosting are close to each other, and both beat recent form. The blend is best on every measure.')
BOX('<b>Why the errors are large.</b> One goal changes a midfielder\'s score by 5 points, so even a perfect model of the <i>chances</i> would have an average error of about 2 points. '
    'The aim is a better ranking than everyone else\'s, not exact scores.')
H2('The bug that fooled us')
P('The first training run looked far better: ranking accuracy of 0.51 for starters, and its top 10 picks averaged 10 points. Then one prediction showed Liverpool playing "MUN (H)" six times in one gameweek.')
P('The cause: <b>fixture IDs restart every season</b>. Fixture 103 exists in all five seasons, so joining team strength on the fixture ID alone copied rows across seasons. '
  'The table had 312,000 rows when it should have had 122,000, and many training rows carried information from other games. After adding the season to the join key, the honest ranking accuracy was 0.25.')
BOX('<b>Lesson.</b> When a result looks too good, look for a leak. Check row counts after every join, and look at a few predictions by eye. '
    'One line of code (a missing key) made a weak model look excellent.')
H2('Other problems we hit')
B(['<b>Segmentation fault on macOS:</b> PyTorch and scikit-learn each load their own copy of the OpenMP threading library. Setting <font face="Courier">KMP_DUPLICATE_LIB_OK=TRUE</font> and <font face="Courier">OMP_NUM_THREADS=1</font> fixes it.',
   '<b>PuLP 4 ships without a solver:</b> we pinned <font face="Courier">pulp==2.9.0</font>, which includes the CBC solver and uses the standard API most tutorials teach.',
   '<b>An illegal squad:</b> a squad with 4 players from one club has no legal solution, so the solver returned nonsense. The optimiser now checks the solver status and the app checks the club limit.'])
story.append(PageBreak())

# ======================================================================= part 5: exercises
H1('Part 5. Exercises and next steps')
P('Work through these in order. Each one teaches one new idea, and you can measure the result with <font face="Courier">train_mlp.py</font>.')
B(['<b>1. Change the size.</b> Try hidden=64 and hidden=256. Does the validation loss improve? This teaches capacity and overfitting.',
   '<b>2. Remove dropout</b> (drop=0). Watch the validation loss stop improving earlier.',
   '<b>3. Use the GPU.</b> Move the model and tensors to "mps" and time one epoch on each device. For a model this small the CPU may win, which is a useful lesson.',
   '<b>4. Use a DataLoader.</b> Replace the manual batching with <font face="Courier">torch.utils.data.TensorDataset</font> and <font face="Courier">DataLoader(batch_size=512, shuffle=True)</font>.',
   '<b>5. Player embeddings.</b> Add <font face="Courier">nn.Embedding(n_players, 8)</font> keyed on the player code. Then plot the learned vectors to see which players the network thinks are alike.',
   '<b>6. Predict a distribution.</b> Replace the single output with probabilities for 0 to 1, 2 to 3, 4 to 6 and 7+ points, using <font face="Courier">nn.CrossEntropyLoss</font>. That lets the app tell a safe pick from a high-ceiling one, which matters for captaincy.',
   '<b>7. A sequence model.</b> Feed each player\'s last 10 games to an <font face="Courier">nn.GRU</font> in place of the rolling averages.'])
H2('After this project')
P('Once these feel familiar, the cricket video idea is the next step: a pretrained detection model (YOLO, built on PyTorch) to find the ball in clips you film yourself, then fine-tuning it. '
  'Fine-tuning is the same training loop from Part 2, run on a model someone else has already trained.')

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title='PyTorch and FPL: study notes', author='Pranav Arakkal')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('built', OUT)
