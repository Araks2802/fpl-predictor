"""Build notes/FPL_Project_Complete_Guide.pdf: the whole project, every step, every decision and every failure.

Numbers come from notes/report_facts.json (made by notes/gather_facts.py) and the saved model results, and code listings are
read straight from the project files, so the report always matches the code.
"""
import inspect, json, os, sys, textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, PageBreak, Image, KeepTogether, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

OUT = ROOT / 'notes' / 'FPL_Project_Complete_Guide.pdf'
FIG = ROOT / 'notes' / 'fig'
F = json.load(open(ROOT / 'notes' / 'report_facts.json'))
ss = getSampleStyleSheet()
NAVY = colors.HexColor('#37003c'); TEAL = colors.HexColor('#0b6a5a'); GREY = colors.HexColor('#555555'); LIGHT = colors.HexColor('#f4f1f6')
S = {
    'title': ParagraphStyle('t', parent=ss['Title'], fontSize=28, leading=33, textColor=NAVY, spaceAfter=8, alignment=0),
    'sub': ParagraphStyle('s', parent=ss['Normal'], fontSize=13, leading=18, textColor=GREY, spaceAfter=14),
    'h1': ParagraphStyle('h1', parent=ss['Heading1'], fontSize=19, leading=23, textColor=NAVY, spaceBefore=4, spaceAfter=10),
    'h2': ParagraphStyle('h2', parent=ss['Heading2'], fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=5),
    'h3': ParagraphStyle('h3', parent=ss['Heading3'], fontSize=11.2, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=3),
    'p': ParagraphStyle('p', parent=ss['Normal'], fontSize=10.2, leading=14.6, spaceAfter=6),
    'b': ParagraphStyle('b', parent=ss['Normal'], fontSize=10.2, leading=14.4, leftIndent=14, bulletIndent=3, spaceAfter=3),
    'code': ParagraphStyle('c', fontName='Courier', fontSize=7.6, leading=9.6, backColor=LIGHT, borderPadding=6, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=10),
    'box': ParagraphStyle('x', parent=ss['Normal'], fontSize=10.2, leading=14.4, backColor=colors.HexColor('#e9f7f1'), borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=12),
    'warn': ParagraphStyle('w', parent=ss['Normal'], fontSize=10.2, leading=14.4, backColor=colors.HexColor('#fdecef'), borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=12),
    'cap': ParagraphStyle('cap', parent=ss['Normal'], fontSize=8.6, leading=11, textColor=GREY, spaceAfter=10),
    'cell': ParagraphStyle('cell', parent=ss['Normal'], fontSize=8.4, leading=10.4),
    'toc1': ParagraphStyle('toc1', fontSize=10.5, leading=15, leftIndent=8),
}
story = []


class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):     # feed chapter headings into the table of contents
        if isinstance(f, Paragraph) and f.style.name == 'h1':
            self.notify('TOCEntry', (0, f.getPlainText(), self.page))


def P(t, s='p'): story.append(Paragraph(t, S[s]))
def H1(t): story.append(CondPageBreak(6 * cm)); story.append(Paragraph(t, S['h1']))
def H2(t): story.append(CondPageBreak(3 * cm)); story.append(Paragraph(t, S['h2']))
def H3(t): story.append(Paragraph(t, S['h3']))
def B(items):
    for t in items: story.append(Paragraph(t, S['b'], bulletText='\u2022'))
def BOX(t): story.append(Paragraph(t, S['box']))
def WARN(t): story.append(Paragraph(t, S['warn']))
def wrap_code(code, width=104):
    out = []
    for line in code.rstrip('\n').split('\n'):
        ind = len(line) - len(line.lstrip())
        out += textwrap.wrap(line, width, subsequent_indent=' ' * (ind + 4), drop_whitespace=False, replace_whitespace=False) or ['']
    return '\n'.join(out)
def C(code): story.append(Preformatted(wrap_code(textwrap.dedent(code).strip('\n')), S['code']))
def SRC(obj, title=None):
    if title: H3(title)
    C(inspect.getsource(obj))
def FIGURE(name, cap, w=15.5):
    from reportlab.lib.utils import ImageReader
    ir = ImageReader(str(FIG / name)); iw, ih = ir.getSize(); w = w * cm
    story.append(KeepTogether([Image(str(FIG / name), width=w, height=w * ih / iw), Paragraph(cap, S['cap'])]))
def T(rows, widths, wrap=True):
    rows = [[Paragraph(str(c), S['cell']) if wrap and i > 0 else c for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTSIZE', (0, 0), (-1, 0), 8.6),
                           ('FONTSIZE', (0, 1), (-1, -1), 8.4), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
                           ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.append(t); story.append(Spacer(1, 8))
def footer(c, d):
    c.saveState(); c.setFont('Helvetica', 8); c.setFillColor(GREY)
    c.drawString(2 * cm, 1.2 * cm, 'FPL Points Predictor: the complete guide'); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'Page {d.page}'); c.restoreState()


sc = pd.DataFrame(F['scores']); st_ = sc[sc.group == 'starters'].set_index('predictor'); al_ = sc[sc.group == 'all'].set_index('predictor')
RB = F['rows_by_season']

bl, l5 = st_.loc['blend'], st_.loc['last5']
# ======================================================================= cover
story += [Spacer(1, 50)]
P('FPL Points Predictor', 'title')
P('The complete guide: every step, every decision, every failure, and the PyTorch behind it', 'sub')
BOX('<b>What this document is.</b> A full record of the project: what we built, why each choice was made, how the data was handled, what every feature means, '
    'how the PyTorch model works line by line, how the transfer optimiser follows the FPL rules, how we tested it honestly, and every bug and dead end along the way. '
    'Read it front to back once, then use it as a reference. Every number comes from the saved results, and every code listing is copied from the project files.')
T([['Fact', 'Value'], ['Seasons of data', '5 (2022-23 to 2026-27 so far)'], ['Player-match rows', f"{F['rows_total']:,} ({F['rows_hist']:,} with results)"],
   ['Features per row', str(F['n_features'])], ['PyTorch network weights', f"{F['n_params']:,} per network, 5 networks"],
   ['Ranking accuracy, regular starters', f"{bl['rank corr (per GW)']:.3f} (recent form: {l5['rank corr (per GW)']:.3f})"],
   ["Avg actual points of each week's top 10 picks", f"{bl['avg pts of top-10 picks']:.1f} (recent form: {l5['avg pts of top-10 picks']:.1f})"],
   ['Transfer search', 'exact (mixed-integer linear programming), under 1 second']], [7.5, 8.5])
story.append(PageBreak())
P('<b>Contents</b>', 'h2')
toc = TableOfContents(); toc.levelStyles = [S['toc1']]; story.append(toc)
story.append(PageBreak())

# ======================================================================= 1
H1('1. The project in one page')
P('<b>The problem.</b> Every week a Fantasy Premier League manager decides who to start, who to captain and who to transfer. Most people decide on <i>form</i> '
  '(points over the last few games) and gut feel. Form is very noisy: a defender who scored twice last month is not a goal threat. '
  'We wanted an honest number for every player, "how many points should we expect from him this week?", and then a tool that turns those numbers into decisions.')
P('<b>The solution, in four layers.</b>')
T([['Layer', 'What it does', 'Main tools'], ['1. Data', 'Downloads five seasons of game-by-game stats and joins them into one table: one row per player per match.', 'requests, pandas'],
   ['2. Prediction', 'Predicts each player\'s points in each upcoming fixture, plus his chance of a goal, an assist, a clean sheet, a defensive contribution, bonus and 60+ minutes.', 'PyTorch, scikit-learn'],
   ['3. Decisions', 'Finds the best legal line-up, captain and transfers under every FPL rule.', 'PuLP (linear programming)'],
   ['4. App', 'A website where a manager enters a team ID and sees everything.', 'Streamlit, Plotly, HTML/CSS']], [2.6, 9.4, 4])
P('<b>The result.</b> On gameweeks the models never saw, the predictions rank regular starters with a rank correlation of '
  f"<b>{bl['rank corr (per GW)']:.3f}</b> against <b>{l5['rank corr (per GW)']:.3f}</b> for recent form. Each week's ten highest-predicted starters went on to average "
  f"<b>{bl['avg pts of top-10 picks']:.1f}</b> points, against {l5['avg pts of top-10 picks']:.1f} for the ten best by form.")
H2('How the pieces connect')
C('''
 FPL API (this season)      GitHub archive (4 past seasons)
          |                            |
          +------------+---------------+
                       v
        fetch_current.py  ->  data/2026-27/*.json, history.csv
                       v
        fpl_data.py   build(): one row per player per fixture, 67 features
                       v
        train_mlp.py     PyTorch ensemble + gradient boosting  ->  models/mlp.pt, gbm.joblib
        train_events.py  chance models (goal, assist, CS, DC, bonus, 60 mins)  ->  models/events.joblib
                       v
        make_predictions.py  ->  data/predictions.parquet   (every player, every remaining gameweek)
                       v
        app.py + ui.py + fpl_engine.py   (Streamlit app: reads predictions, loads a team, runs the optimiser)
''')
P('The app never trains anything. All the heavy work happens offline, and the app only reads one file of predictions. '
  'That keeps the website fast, and means the online version does not even need PyTorch installed.')

# ======================================================================= 2
H1('2. The FPL rules, and what they mean for modelling')
P('You cannot model a game without knowing its rules. These are the 2026/27 rules the tool follows, with a note on how each one shaped the project.')
T([['Rule', 'Detail', 'What it meant for us'], ['Squad', '15 players: 2 GK, 5 DEF, 5 MID, 3 FWD. Budget £100m. Max 3 per club.', 'Hard constraints in the optimiser.'],
   ['Starting XI', '1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD. 4 on the bench.', 'Each week the optimiser picks the best legal formation.'],
   ['Captain', 'Scores double. Vice takes over if the captain plays no minutes. Triple Captain chip: x3.', 'The captain\'s points count twice in the objective.'],
   ['Transfers', '1 free per week, roll up to 5. Each extra costs 4 points.', 'Hits in the objective; free transfers worked out from history.'],
   ['Selling price', 'You keep only half of any price rise, rounded down to £0.1m. Falls are passed on in full.', 'We rebuild every player\'s buy price to know true spending power.'],
   ['Chips', 'Two sets (GW1-19 and GW20-38): Wildcard, Free Hit, Bench Boost, Triple Captain. One per week.', 'Wildcard and Free Hit remove transfer limits in the optimiser.'],
   ['Blank / double GWs', 'A club may play 0 or 2 matches in a gameweek.', 'Predict per fixture, then add up per gameweek.']], [2.6, 7, 6.4])
H2('Scoring')
T([['Action', 'GK', 'DEF', 'MID', 'FWD'], ['Up to 60 min / 60+ min', '1 / 2', '1 / 2', '1 / 2', '1 / 2'], ['Goal', '10', '6', '5', '4'], ['Assist', '3', '3', '3', '3'],
   ['Clean sheet (60+ min)', '4', '4', '1', '0'], ['Every 3 saves', '1', '-', '-', '-'], ['Defensive contribution', '-', '2 (10+ CBIT)', '2 (12+ CBIRT)', '2 (12+ CBIRT)'],
   ['Every 2 goals conceded', '-1', '-1', '-', '-'], ['Bonus (top 3 by BPS in the match)', '3/2/1', '3/2/1', '3/2/1', '3/2/1'],
   ['Yellow / red / own goal / pen miss', '-1/-3/-2/-2', 'same', 'same', 'same']], [6, 2.4, 2.6, 2.6, 2.4])
P('CBIT = clearances, blocks, interceptions, tackles. CBIRT adds recoveries. Defensive contributions began in 2025-26, and 2026-27 changed the bonus system '
  '(for example one BPS point per three CBIs instead of two). <b>Two rule changes in two seasons</b> is why the model needs a way to notice that scoring has shifted (section 4.6).')
H2('Why the scoring makes prediction hard')
FIGURE('points_dist.png', f"Points per player-match across all five seasons. {100 * F['points_dist']['share_0']:.0f}% of rows are 0 points and {100 * F['points_dist']['share_le2']:.0f}% are 2 or fewer; "
       f"only {100 * F['points_dist']['share_ge10']:.1f}% are 10 or more. The maximum is {F['points_dist']['max']}.", 13)
B([f"<b>Most rows are players who did not play.</b> {100 * F['minutes_zero_share']:.0f}% of player-match rows have 0 minutes (squad players, injured, reserves). The model has to learn who will actually play.",
   '<b>Hauls are rare and decide everything.</b> A single goal adds 4 to 10 points, so one moment can turn a 2 into a 9. Exact scores are mostly luck; the realistic aim is a better <i>ranking</i>.',
   '<b>Expected points is an average.</b> A forward with 5.0 expected points will usually score 2, sometimes 9 or 12. That is why we also give chances of a goal, clean sheet and so on.'])

# ======================================================================= 3
H1('3. The data')
H2('3.1 Sources')
T([['Source', 'What we take from it', 'Why'], ['Official FPL API, fantasy.premierleague.com/api/', 'This season: every player\'s game-by-game history, prices, ownership, injury news, chance of playing, fixtures, fixture difficulty, deadlines, and any manager\'s team.', 'The most current data, and the same data the FPL website uses. Public, no key or login.'],
   ['vaastav/Fantasy-Premier-League (GitHub)', 'Seasons 2022-23 to 2025-26: merged_gw.csv (every player, every match), players_raw.csv, fixtures.csv, teams.csv.', 'The API only shows the current season. This archive has saved the same API data weekly for years.'],
   ['FPL image server', 'Shirt images for each club (by team code).', 'Used only in the app, loaded by the browser.']], [4.2, 7, 4.8])
H2('3.2 The API endpoints we call')
T([['Endpoint', 'Gives'], ['bootstrap-static/', 'All players (price, position, team, ownership, form, status, news, chance of playing), teams, gameweeks and deadlines, game settings.'],
   ['fixtures/', 'Every fixture of the season with kick-off time, scores and the official difficulty rating (1 to 5) for each side.'],
   ['element-summary/{id}/', 'One player\'s match-by-match stats this season. fetch_current.py calls it for all ~670 players with a short pause between calls.'],
   ['entry/{id}/, entry/{id}/history/, entry/{id}/event/{gw}/picks/, entry/{id}/transfers/', "A manager's name, rank, bank, chips used, squad and transfer history. Called live by the app."]], [6.5, 9.5])
P('The API is public but not officially documented, so it could change. We fetch politely: once a week, with a 0.05-second pause between player requests.')
H2('3.3 Building one table (fpl_data.build)')
P('Each season file is loaded, reduced to the same columns, and stacked. Then, in order:')
B(['<b>Fix each row\'s team from the fixture</b> (home side if he was at home, away side if not). The season files list a player\'s end-of-season club, which is wrong for anyone who moved mid-season (section 13).',
   '<b>Drop duplicate rows</b> (10 repeated rows in the source files).',
   '<b>Add future rows:</b> one row per player for every upcoming fixture, with no stats, so the same code produces the features we predict from.',
   '<b>Join team strength</b> for his team and the opponent, keyed on <i>season and fixture</i>, because fixture IDs restart every season (section 13, the big bug).',
   '<b>Sort by player and time</b> and compute all the rolling features, always shifted by one game so a match never sees its own result.',
   '<b>Hold form for future weeks:</b> every future row gets the player\'s form as of today, so projections for later weeks do not drift.'])
T([['Season', 'Player-match rows (with results)', 'Players']] + [[s, f'{RB[s]:,}', str(F['players_by_season'][s])] for s in RB], [4, 6, 4])
P('2024-25 is slightly lower than the raw file because that season included "assistant manager" entries (position 5), which we exclude.')
H2('3.4 The target')
P('The model predicts <b>total_points</b> for a player in one fixture: the official FPL score, which already includes appearance points, goals, assists, clean sheets, '
  'saves, defensive contributions, bonus and deductions. Separate models predict the individual events (chapter 8).')

# ======================================================================= 4 features
H1('4. Every feature, and why it is there')
P('A <b>feature</b> is one number the model gets for a row (a player in a fixture). All 67 are computed only from information available <b>before kick-off</b>. '
  'The rule is enforced in code: every rolling feature uses <font face="Courier">shift(1)</font>, which moves the history down one row so a match never sees its own result.')
C('''
# the pattern used for every rolling feature (fpl_data.py)
h[f'{c}_r{w}'] = gcode[c].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
#   gcode     = rows grouped by player, in time order
#   shift(1)  = look only at earlier games
#   rolling(w)= the last w of those games;  min_periods=1 = use what exists early on
''')
DESC = {
    'pos': ('Position (1 GK, 2 DEF, 3 MID, 4 FWD)', 'Scoring rules differ by position. The network also gets it as an embedding (6.4).'),
    'price': ('Current price in £m', 'Price is the market\'s summary of quality and role.'),
    'home': ('1 if at home', 'Teams score more at home.'),
    'own_att': ('His team\'s goals per game over the last 8, shrunk to the league mean', 'Attacking strength of his team.'),
    'own_def': ('His team\'s goals conceded per game over the last 8, shrunk', 'Defensive strength (clean sheets, goals conceded).'),
    'opp_att': ('Opponent\'s goals per game over the last 8, shrunk', 'How dangerous the opponent is.'), 'opp_def': ('Opponent\'s goals conceded per game, shrunk', 'How easy the opponent is to score against.'),
    'dc_era': ('1 from 2025-26 (defensive contributions exist)', 'Tells the model when DC points were possible.'),
    'n_fix_gw': ('Fixtures his team has in this gameweek', 'Flags doubles (rows are per fixture, so mostly 1).'), 'games_seen': ('Number of earlier rows for him', 'How much history stands behind his averages.'),
    'start_rate5': ('Share of his last 5 games with 60+ minutes', 'Nailed starter or rotation risk.'), 'season_pts_mean': ('His average points so far this season', 'Current-season level.'),
    'prev_ppg': ('Last season\'s points per appearance', 'Longer-term level; helps early in a season.'), 'prev_xgi90': ('Last season\'s xG + xA per 90', 'Longer-term attacking threat.'),
    'pts_p90_r10': ('Points per 90 minutes over the last 10 games (needs 270+ minutes)', 'Separates quality from minutes.'), 'xgi_p90_r10': ('xG + xA per 90 over the last 10', 'Attacking threat per minute.'),
    'pts_p90_r30': ('Points per 90 over the last 30', 'A steadier version.'), 'xgi_p90_r30': ('xG + xA per 90 over the last 30', 'A steadier version.'),
    'season_xgi_mean': ('Average xG + xA per game this season', 'Current-season threat.'),
    'own_xgf': ('His team\'s expected goals per game over the last 10 played, shrunk', 'Attack strength without finishing luck.'),
    'own_xga': ('His team\'s expected goals against per game, shrunk', 'Defence strength without luck (Arsenal\'s defence lives here).'),
    'opp_xgf': ('Opponent\'s xG per game, shrunk', 'Threat to his clean sheet.'), 'opp_xga': ('Opponent\'s xG against per game, shrunk', 'How many chances his team should get.'),
    'att_vs_def': ('own_xgf x opp_xga / league mean', 'Expected goals for his team in this match.'),
    'cs_chance': ('exp(-opp_xgf x own_xga / league mean)', 'Rough clean-sheet chance: probability of 0 goals in a Poisson model.'),
    'pos_level': ('League-wide average points of 60+ minute players in his position over the previous 6 gameweeks', 'Lets the model notice rule changes that shift scoring.'),
    'own_form5': ('His team\'s league points per game over the last 5, shrunk towards 1.37', 'Team form (Brighton\'s hot run lives here).'), 'opp_form5': ('Opponent\'s points per game over the last 5', 'Opponent form.'),
    'own_gd5': ('His team\'s goal difference per game over the last 5, shrunk', 'Team form by margin.'), 'opp_gd5': ('Opponent\'s goal difference per game', 'Opponent form by margin.'),
    'h2h_adj': ('His points against this opponent minus his usual, shrunk (K=4)', 'Does he do better against this club?'), 'h2h_n': ('Games he has played against this opponent', 'How much to trust h2h_adj.'),
    'venue_adj': ('His points at home (or away) minus his usual, shrunk', 'Home/away player effect.'),
    'dc_hit_r5': ('Share of his last 5 games (with data) reaching his DC threshold', 'Defensive-contribution habit.'), 'dc_hit_r10': ('Same over the last 10', 'Steadier DC habit.'),
    'dc_p90_r10': ('Defensive actions per 90 over the last 10 (270+ min)', 'How close he usually gets to the threshold.'),
    'bonus_rate_r10': ('Share of his last 10 played games with bonus', 'Bonus habit.'), 'bps_p90_r10': ('Bonus-system points per 90 over the last 10', 'Underlying bonus score.'),
}
NAMES = {'total_points': 'points', 'minutes': 'minutes', 'bps': 'bonus-system points', 'expected_goals': 'xG', 'expected_assists': 'xA',
         'defensive_contribution': 'defensive actions', 'saves': 'saves', 'bonus': 'bonus points'}
for w in (3, 5, 10, 20):
    for c, nm in NAMES.items():
        DESC.setdefault(f'{c}_r{w}', (f'Average {nm} over his last {w} games', 'Recent form' + (' (short window reacts fast, long window is steadier)' if w in (3, 20) else '.')))
import fpl_data as fd
missing = [f for f in fd.FEATURES if f not in DESC]
assert not missing, f'features without a description: {missing}'
groups = [('Base features (the first version)', fd.BASE_FEATURES), ('Longer form, per-90 rates, last season, xG team strength, scoring level', fd.EXTRA_FEATURES),
          ('Team form', fd.TEAM_FORM), ('Matchup: opponent and home/away', fd.MATCHUP), ('Defensive contributions and bonus', fd.DC_BONUS)]
for title, cols in groups:
    H3(f'{title} ({len(cols)})')
    T([['Feature', 'Meaning', 'Why']] + [[f'<font face="Courier" size="7.5">{c}</font>', DESC[c][0], DESC[c][1]] for c in cols], [4.2, 6.6, 5.2])
H2('4.1 Shrinkage: why so many features are "pulled towards the average"')
P('A team that conceded 0 goals in its first 2 games is not a perfect defence. Small samples swing wildly. <b>Shrinkage</b> mixes a raw average with the league average, '
  'weighted by how much data there is:')
C('''
shrunk = (sum of the team's values + K x league_mean) / (number of games + K)
# K is like K "imaginary average games". With 2 real games and K=6, the result is mostly league average;
# with 8 real games it is mostly the team's own number.
''')
P('We use K=6 for goals-based strength, K=5 for xG strength, 2 games for team form, and K=4 for the opponent and home/away effects (which are the noisiest).')
H2('4.2 Why expected goals (xG)?')
P('xG is the chance a shot becomes a goal, judged from where and how it was taken. Summed over a match it says how many goals a team <i>deserved</i>. '
  'Actual goals contain finishing luck that does not repeat; xG is steadier, so it predicts the next game better. We give the model both, and it decides how much to trust each.')
H2('4.3 Missing values')
P('Some features do not exist for a row: a new player has no last season, per-90 rates need 270 minutes, and defensive contributions were not recorded before 2025-26. '
  'We leave these as missing (NaN) rather than 0, because 0 would be a false statement ("he made no defensive actions"). '
  'For the neural network missing becomes 0 <i>after</i> standardising, which means "average". Gradient boosting gets -1, a value it can learn to treat separately.')
H2('4.4 Holding form for future weeks')
P('Future fixtures have no results. If we let the rolling windows run on, a window of 3 would be empty after 3 future weeks and every player would drift to "average". '
  'We spotted this when the season-long totals dropped by a third after Gameweek 13. The fix copies each player\'s form, as of today, into every future row.')
H2('4.5 Team codes, not team IDs')
P('FPL numbers teams 1 to 20 each season in alphabetical order, so promotion and relegation change the numbers. Following "team 10" across seasons would mix clubs. '
  'Every team calculation therefore uses the club\'s permanent <b>code</b>.')
H2('4.6 The scoring level feature')
P('In 2025-26 defenders started earning defensive-contribution points, and in 2026-27 the bonus system changed. A model trained mostly on older seasons would '
  'misjudge the new scoring. <font face="Courier">pos_level</font> gives the league-wide average for each position over the previous 6 gameweeks, so the model can see the shift as it happens.')

# ======================================================================= 5 ML basics
H1('5. Machine learning, the parts you need')
H2('5.1 Supervised regression')
P('We have many examples of (features, answer): the features of a player before a match, and the points he then scored. A model is a function with adjustable numbers '
  '(<b>parameters</b> or <b>weights</b>). <b>Training</b> adjusts the weights so the function\'s outputs get close to the real answers. Because the answer is a number, this is <b>regression</b>. '
  'The chance models (goal or not) are <b>classification</b>.')
H2('5.2 Loss: how wrong is the model?')
P('The <b>loss</b> is one number summarising the errors. We use <b>mean squared error (MSE)</b>: the average of (prediction - actual) squared. Squaring punishes big misses and, '
  'importantly, the prediction that minimises MSE is the <b>average</b> outcome. That is exactly "expected points".')
H2('5.3 Train, validation and test sets')
T([['Set', 'Rows', 'Used for'], ['Train', f"{F['split']['train']:,}: 2022-23, 2023-24, 2024-25, and 2025-26 GW1-28", 'Learning the weights.'],
   ['Validation', f"{F['split']['val']:,}: 2025-26 GW29-38", 'Deciding when to stop training (early stopping) and comparing settings.'],
   ['Test', f"{F['split']['test']:,}: 2026-27 GW3-10", 'The final, honest score. Never used for any decision during training.']], [2.4, 8, 5.6])
P('The split is <b>by time</b>. A random split would put, say, Gameweek 30 in training and Gameweek 29 in testing, so the model would learn from the future. '
  'In real use we always predict a week we have not seen, so the test must do the same. After testing, the final model is retrained on everything played so far.')
H2('5.4 Overfitting')
P('A flexible model can memorise the training rows, including their luck, and then predict new matches badly. The sign is training loss falling while validation loss stops falling or rises.')
FIGURE('training_curve.png', f"One real training run. Training loss keeps falling for 30 epochs; validation loss is best at epoch {F['curve']['best_epoch']} and then drifts up. "
       'Early stopping keeps the weights from the best validation epoch. (Validation loss is lower than training loss here because the validation weeks are a different, '
       'slightly lower-scoring period and dropout is switched off when validating.)', 14.5)
H2('5.5 How we score models')
T([['Measure', 'What it means', 'Good is'], ['RMSE (typical error)', 'Square root of MSE, in points. Big misses count a lot.', 'Lower'],
   ['MAE (average error)', 'Average size of the miss, in points.', 'Lower'],
   ['Ranking accuracy', 'Spearman rank correlation between predicted and actual points within each gameweek, averaged. 1 = perfect order, 0 = random.', 'Higher. This matters most for FPL decisions.'],
   ['Top-10 picks', "Each week, the actual average points of the 10 highest-predicted regular starters.", 'Higher'],
   ['Calibration', 'Do predictions match reality on average, overall and for groups (by position, by prediction level)?', 'Predicted close to actual'],
   ['Brier skill', 'For chance models: how much smaller the squared error is than guessing the average rate for everyone.', 'Higher than 0%']], [3.4, 9.4, 3.2])
P('"Regular starters" (45+ minutes a game over the last 3) is the group that matters for decisions. Including every bench player makes any model look good, because predicting 0 for someone who never plays is easy.')

# ======================================================================= 6 PyTorch
H1('6. PyTorch: why, what, and every line of our model')
H2('6.1 Why PyTorch?')
B(['<b>To learn it.</b> PyTorch is the most widely used deep-learning library in research and industry. Learning it on data you understand is the fastest way in.',
   '<b>Flexibility.</b> You write the model as ordinary Python, so ideas like the position embedding, a multi-task model or a sequence model are easy to add.',
   '<b>Honesty about the trade-off.</b> On tabular data like ours, gradient boosting is about as good (chapter 7). We kept both and averaged them, which beat either alone. '
   'That is a perfectly good answer in an interview: you chose the tool to learn, measured it against a strong baseline, and combined them.'])
H2('6.2 The four building blocks')
T([['Block', 'What it is', 'In our project'], ['Tensor', 'An array of numbers that can sit on a GPU and remember how it was computed.', 'Features become a float32 tensor of shape (rows, 67); positions an int64 tensor.'],
   ['Autograd', 'PyTorch records every operation and can compute the gradient of the loss with respect to every weight with one call: loss.backward().', 'Used in every training step.'],
   ['nn.Module', 'A class holding layers (and their weights) plus a forward() saying how data flows.', 'PointsNet.'],
   ['Optimiser', 'Uses the gradients to update the weights.', 'AdamW, learning rate 0.001, weight decay 0.0001.']], [2.4, 7.4, 6.2])
H2('6.3 Gradient descent in 6 lines')
C('''
w = torch.tensor(3.0, requires_grad=True)     # one weight
for step in range(50):
    loss = (w - 5) ** 2                       # smallest when w = 5
    loss.backward()                           # d(loss)/dw, here 2(w-5) = -4 at the start
    with torch.no_grad(): w -= 0.1 * w.grad   # step against the gradient (0.1 = learning rate)
    w.grad.zero_()                            # gradients add up unless reset
# w ends at 5.0. Training a network is this, with 18,081 weights instead of 1.
''')
H2('6.4 Our network: PointsNet')
import fpl_model
SRC(fpl_model.PointsNet)
P('Line by line:')
B(['<font face="Courier">nn.Embedding(4, 8)</font>: a table of 4 rows (GK, DEF, MID, FWD), each a vector of 8 learned numbers. The model invents its own description of each position.',
   '<font face="Courier">nn.Linear(n_feat + 8, 128)</font>: 67 features plus 8 position numbers = 75 inputs, each of 128 outputs is a weighted sum of all 75 plus a constant.',
   '<font face="Courier">nn.ReLU()</font>: turns negatives into 0. This non-linearity lets the network learn curves and interactions ("xG matters more for forwards"); without it, stacked Linear layers collapse into one straight line.',
   '<font face="Courier">nn.Dropout(0.2)</font>: during training, randomly zeroes 20% of the values, so no single path can memorise luck. Switched off at prediction time.',
   'A second hidden layer of 64, then <font face="Courier">nn.Linear(64, 1)</font> outputs one number: expected points.',
   '<font face="Courier">torch.cat([x, self.pos(pos)], dim=1)</font> glues the features and the position vector side by side; <font face="Courier">squeeze(1)</font> turns shape (rows, 1) into (rows,).'])
H3('Counting the weights')
T([['Part', 'Calculation', 'Weights'], ['Embedding', '4 x 8', '32'], ['Hidden layer 1', '(67 + 8) x 128 + 128', '9,728'], ['Hidden layer 2', '128 x 64 + 64', '8,256'], ['Output', '64 x 1 + 1', '65'],
   ['Total', '', f"{F['n_params']:,}"]], [4, 7, 3])
H2('6.5 Preparing inputs: the Scaler')
SRC(fpl_model.Scaler)
P('Each feature is standardised: (value - mean) / standard deviation, using the <b>training</b> rows only. Networks train badly when one input is around 6 (price) and another around 70 (minutes). '
  'Values are clipped to between -6 and 6 so one extreme row cannot dominate, and missing values become 0, the average.')
H2('6.6 The training loop, line by line')
import train_mlp
SRC(train_mlp.fit_nets)
B(['<b>Batches of 512.</b> One gradient from 512 rows is a good estimate and far faster than using all 105,000 rows for every step. <font face="Courier">torch.randperm</font> shuffles the order each epoch.',
   '<b>The five steps:</b> zero_grad (clear old gradients), forward pass, loss, backward (gradients), step (update). Every PyTorch training loop has these five.',
   '<b>net.train() / net.eval()</b> switch dropout on and off. <b>torch.no_grad()</b> skips gradient bookkeeping when only predicting.',
   '<b>Early stopping:</b> after each epoch we check validation MSE. We keep a copy of the best weights and stop after 5 epochs without improvement (at most 40 epochs).',
   '<b>AdamW</b> adapts the step size for each weight from its recent gradients, and weight decay gently pulls weights towards 0 (another defence against overfitting).',
   '<b>Five seeds:</b> each network starts from different random weights and sees batches in a different order, so each makes different mistakes. Averaging them (an <b>ensemble</b>) is steadier than any one.'])
H2('6.7 Predicting, saving and loading')
SRC(fpl_model.predict)
P('The final model\'s weights, the scaler\'s means and standard deviations and the feature list are saved together in models/mlp.pt with '
  '<font face="Courier">torch.save</font>, and loaded with <font face="Courier">fpl_model.load()</font>. We save the <font face="Courier">state_dict</font> (just the weights), which is the recommended way.')
H2('6.8 CPU or GPU?')
P('PyTorch can use Apple\'s GPU through "mps". Our network is tiny (18,081 weights), so the CPU is fast enough and avoids moving data back and forth. '
  'Bigger models (images, video, transformers) are where the GPU pays off.')

# ======================================================================= 7 GBM and blend
H1('7. Gradient boosting, and why we blend')
P('<b>Gradient boosting</b> builds many small decision trees one after another; each new tree corrects the errors of the ones before. Trees split rows on questions like '
  '"is xG per 90 above 0.4?" and handle missing values and odd scales naturally. On tabular data it is usually the strongest single method.')
C('''
HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0)
# 300 trees, each a small step (0.05), each with at most 24 leaves, with a penalty on extreme leaf values
''')
P('The final prediction is <b>(PyTorch ensemble + gradient boosting) / 2</b>. The two models learn differently (smooth functions vs step-like splits), so their errors partly cancel.')
FIGURE('methods.png', 'Regular starters, 2026-27 GW3-10 (never seen). Both models beat recent form clearly; the blend is best or joint best on both measures.', 15)
T([['Method', 'RMSE', 'MAE', 'Ranking accuracy', 'Top-10 picks']] +
  [[{'last5': 'Average of last 5 games', 'gbm': 'Gradient boosting', 'mlp': 'PyTorch ensemble', 'blend': 'Blend (used)'}[k], f'{r.RMSE:.3f}', f'{r.MAE:.3f}', f"{r['rank corr (per GW)']:.3f}", f"{r['avg pts of top-10 picks']:.2f}"]
   for k, r in st_.iterrows()], [5, 2.4, 2.4, 3.4, 2.8])

# ======================================================================= 8 event models
H1('8. The chance models: goal, assist, clean sheet, defensive contribution, bonus, 60 minutes')
P('Expected points is one number. Managers also want to know <i>how</i> a player gets them: is Haaland likely to score, is Raya likely to keep a clean sheet? '
  'These are yes/no events, so each gets a <b>classifier</b> that outputs a probability.')
T([['Event', 'Target (what counts as yes)', 'Rows it learns from', 'Model'], ['Goal', '1+ goals in the match', 'All', 'Gradient-boosting classifier'],
   ['Assist', '1+ assists', 'All', 'Gradient-boosting classifier'], ['60+ minutes', 'Played 60 minutes or more', 'All', 'Gradient-boosting classifier'],
   ['Bonus', '1+ bonus points', 'All', 'Classifier, plus a regressor for how many (1 to 3) when he gets some'],
   ['Defensive contribution', '10+ CBIT (DEF) or 12+ CBIRT (MID, FWD)', 'Outfield players, 2025-26 on only (not recorded before)', 'Gradient-boosting classifier'],
   ['Clean sheet', 'His team concedes 0 and he plays 60+', 'Team-level: one row per team per match', 'Team classifier x his 60+ minute chance']], [2.6, 4.6, 4.6, 4.2])
H2('8.1 Why clean sheets are modelled per team')
P('A clean sheet is a team event: either the whole defence keeps one or nobody does. The first, player-level model under-predicted clean sheets for starters badly '
  '(17% predicted against 27% actual). The team model uses both sides\' xG for and against, recent form, the Poisson clean-sheet chance and the league\'s current clean-sheet rate, '
  'then multiplies by the player\'s chance of playing 60 minutes.')
P('Even so, clean sheets are running hot this season: 29% of team games, against 21% to 27% in the previous four seasons and about 23% expected from the chances teams allow. '
  'Over 200 team games, part of that is luck, so the model expects some return to normal rather than chasing it.')
H2('8.2 Why expected bonus is split in two')
P('The first expected-bonus model (a single regression) gave Doku 1.12 expected bonus points with only a 23% chance of any bonus. That would need almost 5 bonus points '
  'when he gets some, but the maximum is 3. The fix: expected bonus = chance of any bonus x expected size when he gets some (clipped to 1 to 3). The two numbers are now consistent.')
H2('8.3 Double gameweeks and injuries')
C('''
P(at least one in the gameweek) = 1 - (1 - p_fixture1) x (1 - p_fixture2)
then x availability:  chance of playing next week (FPL flag), easing halfway back to fit for each later week
''')
H2('8.4 How well calibrated are they?')
ev = pd.DataFrame(F['events'])
evn = {'goal': 'Goal', 'assist': 'Assist', 'cs': 'Clean sheet (player-level)', 'mins60': '60+ minutes', 'bonus': 'Bonus', 'dc': 'Defensive contribution'}
T([['Event', 'Predicted rate', 'Actual rate', 'Starters pred.', 'Starters actual', 'Top 200 pred.', 'Top 200 actual', 'Skill vs average']] +
  [[evn[r['event']], f"{100 * r['predicted rate']:.1f}%", f"{100 * r['actual rate']:.1f}%", f"{100 * r['starters predicted']:.1f}%", f"{100 * r['starters actual']:.1f}%",
    f"{100 * r['top 200 predicted']:.1f}%", f"{100 * r['top 200 actual']:.1f}%", f"{r['skill vs average %']:.0f}%"] for _, r in ev.iterrows()], [3.4, 1.8, 1.8, 1.9, 1.9, 1.9, 1.9, 1.8])
P('Test period 2026-27 GW3-10. Goals, bonus and defensive contributions are well calibrated; assists are slightly high for the top players; the player-level clean-sheet row shows '
  'the under-prediction that led to the team model. Skill is low for rare events in absolute terms (one match is mostly luck), but every model beats guessing the average.')

# ======================================================================= 9 testing
H1('9. Testing honestly')
H2('9.1 Is the model biased?')
c = F['calib']
FIGURE('calibration.png', f"Calibration on the test weeks. Each dot is 10% of rows grouped by prediction; the dashed line is perfect. Overall the model predicts "
       f"{F['calib_overall'][0]:.2f} points per player-match against an actual {F['calib_overall'][1]:.2f}.", 8.5)
P('You felt the model under-predicted. We checked rather than guessed:')
B(['<b>Overall:</b> almost perfectly calibrated.',
   '<b>Regular starters this season:</b> slightly under-predicted, mostly goalkeepers and forwards (about 0.45 points a game). That is consistent with the 2026-27 bonus changes, which favour exactly those positions.',
   '<b>The whole of 2025-26</b> (trained on the three seasons before): the model <i>over</i>-predicted starters (3.40 against 3.15). So the bias flips between seasons: there is no lasting under-prediction.',
   '<b>Why the numbers still feel low:</b> FPL\'s own "expected points" this early in the season is simply recent form (we checked: the two columns were identical). Form makes hot players look like 10-point players.'])
H2('9.2 Second opinion: a whole unseen season')
P('Eight test gameweeks is a small sample, so key decisions were re-checked by training on 2022-23 to 2024-25 and scoring all 38 gameweeks of 2025-26. '
  'Ranking accuracy there was 0.257 to 0.261 depending on the feature set, close to the 2026-27 result, so the improvement over form holds.')
H2('9.3 What did not help (negative results are results)')
FIGURE('form_weight.png', 'Mixing in extra raw form made predictions worse in both test periods. Hot streaks mostly fade, and the model already uses form at the weight the data supports.', 12)
T([['Idea', 'Test', 'Result', 'Decision'], ['More weight on recent form', '0 to 50% raw form mixed in', 'Ranking fell from 0.269 to 0.229 (2026-27) and 0.261 to 0.217 (2025-26)', 'Default 0%, optional "form boost" slider for users'],
   ['Team form (results, goal difference)', 'With vs without, two test periods', '2025-26: 0.257 both ways; 2026-27: slightly lower', 'Kept (harmless, shown in app). xG strength already carries it.'],
   ['Player vs this opponent, home/away record', 'With vs without', 'No change', 'Kept, shown as a talking point, clearly labelled'],
   ['Defensive contribution and bonus features', 'With vs without', 'Ranking 0.263 both; MAE and top-10 picks slightly better', 'Kept'],
   ['Longer windows, per-90, last season, xG strength', 'With vs without', 'Ranking 0.258 to 0.273, top-10 picks 4.33 to 5.04', 'Kept: the biggest single improvement']], [3.6, 3.2, 5.4, 3.8])

# ======================================================================= 10 optimiser
H1('10. The transfer optimiser')
P('Predictions alone do not tell you what to do. Picking "the 15 best players" breaks the budget and the 3-per-club rule; swapping your worst player for the best one ignores money and positions. '
  'The rules interact, so we use <b>mixed-integer linear programming (MILP)</b>: describe the choices as yes/no variables, the rules as linear constraints, '
  'and the goal as a linear objective. A solver (CBC, through PuLP) then finds the <b>provably best</b> answer.')
H2('10.1 Variables')
T([['Variable', 'Meaning'], ['x[i]', '1 if player i is in the squad after transfers'], ['s[i, g]', '1 if player i starts in gameweek g'],
   ['c[i, g]', '1 if player i is captain in gameweek g'], ['hits', 'Number of extra transfers paid for (integer)'], ['used_ft', 'Free transfers used']], [4, 12])
H2('10.2 Constraints (the FPL rules)')
C('''
sum of x over goalkeepers = 2, defenders = 5, midfielders = 5, forwards = 3
sum of x over each club <= 3
cost of players bought <= bank + selling prices of players sold
each gameweek: 11 starters; 1 GK; 3-5 DEF; 2-5 MID; 1-3 FWD; exactly 1 captain
s[i, g] <= x[i]      (only squad players can start)
c[i, g] <= s[i, g]   (only starters can captain)
transfers in <= max transfers;  hits >= transfers in - free transfers
''')
H2('10.3 Objective')
C('''
maximise   sum over gameweeks g of  weight[g] x sum over players of
               xp[i, g] x ( s[i, g] + c[i, g] + 0.1 x (x[i] - s[i, g]) )
           - 4 x hits  -  roll_value x used_ft
''')
B(['<b>s + c</b>: a starter counts once and the captain twice.', '<b>0.1 x bench</b>: bench players are worth a little (they cover for non-players).',
   '<b>weight[g] = 0.85^k</b>: next week 1.0, then 0.85, then 0.72. Later predictions are less certain and plans change.',
   '<b>roll_value</b> (default 1 point): a saved free transfer has value, so the optimiser will not make a 0.2-point transfer just because it is free. It is 0 when you already have 5.',
   '<b>Wildcard / Free Hit</b>: no transfer limit and no hits; Free Hit plans one week only.'])
H2('10.4 Money: selling prices and free transfers')
import fpl_engine
SRC(fpl_engine.sell_price, 'The selling-price rule')
P('To get each player\'s buy price we take his most recent transfer in from the manager\'s history (ignoring Free Hit weeks, whose squads are temporary). '
  'For players never transferred in, we use his price in the week the manager joined.')
SRC(fpl_engine.free_transfers, 'Replaying free transfers')
P('Wildcard and Free Hit weeks use no free transfers, and saved transfers are kept, matching the 2026/27 rules. The app lets the user correct the number if it differs from the FPL site.')
H2('10.5 Best XI each week')
SRC(fpl_engine.best_xi)
P('A small version of the same idea for one week and a fixed squad. Captain and vice are the two highest projections in the XI; the bench is ordered by projection, goalkeeper first.')
H2('10.6 Safety checks')
P('If no legal squad exists (for example a hand-picked squad with 4 players from one club), the solver cannot succeed. The first version returned whatever values it had, which looked like nonsense. '
  'The optimiser now checks the solver status and raises a clear error, and the app checks the club limit before calling it.')

# ======================================================================= 11 app
H1('11. The app')
T([['Tab', 'What it shows'], ['My team', 'Projected GW points (best XI, captain doubled), captain, bench points, value, bank, free transfers; a pitch with shirts, points and fixtures (hover for chances); next 6 weeks chart; fixture grid; squad table (next GW or next 3).'],
   ['Transfers', 'The optimiser\'s best moves over 1 or 3 gameweeks, with or without Wildcard / Free Hit; every option from hold to 5 transfers; the new team on a pitch.'],
   ['Planner', 'Try your own transfers: pick who to sell and who to buy; see net points after hits, money left, rule problems and a verdict.'],
   ['Captain', 'Captain options from your XI with fixture and ownership; best captains in the game.'],
   ['Differentials', 'Low-owned players ranked by edge = projected points x share of managers who do not own him.'],
   ['Compare', 'Up to 4 players side by side across 24 rows, best in each row in green, plus a per-gameweek chart.'],
   ['Players', 'Every player with filters (position, team, price, ownership, fitness, search) and sorts (points, value, goal / CS / DC / bonus chance...).'],
   ['Fixtures', 'Difficulty grid for every team, and a team-form table (points, goal difference, xG for and against, clean-sheet chance).'],
   ['How it works', 'The method and the test results in plain words.']], [2.8, 13.2])
H2('11.1 How it is built')
B(['<b>Streamlit</b> turns a Python script into a web page. Every click re-runs the script from the top, so slow work is cached with <font face="Courier">@st.cache_data</font> (loading files, loading a team, each optimiser run, each line-up).',
   '<b>ui.py</b> holds the look: CSS (FPL purple, neon green and cyan; Outfit and Inter fonts from Google Fonts) and HTML builders for the pitch, player tables, fixture grid and cards.',
   '<b>Shirts</b> are loaded by the browser from the FPL image server using each club\'s code. Nothing is stored in the project.',
   '<b>fpl_engine.py</b> is shared by the app and the scripts: loading a team, prices, free transfers, best XI and the optimiser.',
   '<b>test_app.py</b> uses Streamlit\'s testing tool to run every tab with no team, a team ID and a hand-picked team, plus chips, horizons and planner cases, and reports any error.'])
H2('11.2 Things we decided not to build')
P('<b>Automatic transfers on a user\'s real FPL team.</b> A team ID only allows reading. Changing a team needs the user\'s FPL login; FPL offers no official way to authorise an app, so we would have to store passwords, '
  'and automated logins break FPL\'s terms and its bot protection. A failure 30 minutes before a deadline could cost a manager points. Instead: a ready-to-copy checklist and a late refresh with team news (planned after deployment).')

# ======================================================================= worked example (computed live from the saved models)
import joblib, numpy as np, fpl_model as fm, fpl_engine as fe2
H1('12. Worked example, one prediction from start to finish: Haaland, Gameweek 11')
P('Everything above is easier to understand on one real example. These numbers are computed from the saved models while this report is built.')
hh = fd.build(horizon=3); Pp = pd.read_parquet(ROOT / 'data' / 'predictions.parquet')
PID = 411; code = int(Pp.loc[PID, 'code']); g0 = int(json.load(open(ROOT / 'data' / 'meta.json'))['gws'][0])
past = hh[(hh.code == code) & ~hh.future].tail(5)
TN = {(t['id']): t['short_name'] for t in json.load(open(ROOT / 'data' / '2026-27' / 'bootstrap.json'))['teams']}
opp_name = lambda r: (TN.get(int(r.opponent_team), str(int(r.opponent_team))) if r.season == '2026-27' else str(int(r.opponent_team))) + (' (H)' if r.was_home else ' (A)')
H2('Step 1. His last five matches (the raw data)')
T([['GW', 'Opponent', 'Minutes', 'Goals', 'Assists', 'xG', 'xA', 'Bonus', 'Points']] +
  [[int(r.gw), opp_name(r), int(r.minutes), int(r.goals_scored), int(r.assists), f'{r.expected_goals:.2f}', f'{r.expected_assists:.2f}', int(r.bonus), int(r.total_points)] for r in past.itertuples()],
  [1.4, 2, 1.8, 1.6, 1.8, 1.6, 1.6, 1.6, 1.6])
row = hh[(hh.code == code) & hh.future & (hh.gw == g0)].iloc[[0]]
H2('Step 2. The feature row for his next fixture')
KEY = ['price', 'home', 'total_points_r5', 'minutes_r5', 'expected_goals_r5', 'expected_goals_r20', 'xgi_p90_r10', 'start_rate5', 'prev_ppg', 'own_xgf', 'opp_xga', 'att_vs_def', 'own_form5', 'opp_form5', 'pos_level', 'h2h_adj']
nets, scl = fm.load(ROOT / 'models' / 'mlp.pt'); gbm = joblib.load(ROOT / 'models' / 'gbm.joblib')
zs = ((row[scl.F] - scl.mu) / scl.sd).iloc[0]
T([['Feature', 'Value', 'Standardised', 'Read it as']] +
  [[f'<font face="Courier" size="7.5">{k}</font>', f'{row[k].iloc[0]:.2f}', f'{zs[k]:+.2f}', 'well above average' if zs[k] > 1 else ('above average' if zs[k] > 0.3 else ('about average' if zs[k] > -0.3 else 'below average'))] for k in KEY],
  [4.6, 2.2, 4.6, 4.6])
P('Standardised values say how many standard deviations above (+) or below (-) the average training row each number is. Most rows are bench players, so a regular starter like Haaland is far above average on minutes and xG.')
H2('Step 3. The models')
x, ps, _ = scl.tensors(row)
import torch
with torch.no_grad(): each = [float(n(x, ps).numpy()[0]) for n in nets]
gbv = float(gbm.predict(row[getattr(gbm, 'feature_list_', fd.FEATURES)].fillna(-1))[0]); mlpv = float(np.mean(each)); blend = (mlpv + gbv) / 2
T([['Model', 'Expected points']] + [[f'PyTorch network {k + 1}', f'{v:.2f}'] for k, v in enumerate(each)] +
  [['PyTorch ensemble (average of 5)', f'{mlpv:.2f}'], ['Gradient boosting', f'{gbv:.2f}'], ['Blend = (ensemble + boosting) / 2', f'<b>{blend:.2f}</b>']], [9, 4])
ch = Pp.loc[PID, 'chance']
P(f"The five networks disagree a little, which is why we average them. His FPL availability flag is {'none (fit)' if pd.isna(ch) else f'{int(ch)}%'}, so the final projection is "
  f"<b>{Pp.loc[PID, f'xp_{g0}']:.2f}</b> points (the app shows the same number).")
H2('Step 4. His chances')
T([['Event', 'Chance'], ['Scores', f"{100 * Pp.loc[PID, f'p_goal_{g0}']:.0f}%"], ['Assists', f"{100 * Pp.loc[PID, f'p_assist_{g0}']:.0f}%"], ['Gets bonus', f"{100 * Pp.loc[PID, f'p_bonus_{g0}']:.0f}%"],
   ['Expected bonus points', f"{Pp.loc[PID, f'xb_{g0}']:.2f}"], ['Plays 60+ minutes', f"{100 * Pp.loc[PID, f'p_60_{g0}']:.0f}%"]], [7, 4])
P('A rough check that these fit together: 2 appearance points x 0.9 + a 4-point goal x 0.57 + 3 points for an assist x 0.13 + 1.13 expected bonus points comes to roughly 5.6, '
  f"close to the model's {Pp.loc[PID, f'xp_{g0}']:.1f}. The points model and the chance models are separate, so they will never agree exactly, but they tell the same story.")
H2('Step 5. A real transfer decision')
boot = json.load(open(ROOT / 'data' / '2026-27' / 'bootstrap.json'))
try:
    tm1 = fe2.load_entry(1, boot); ids = [int(i) for i in tm1['squad'].id]; gw3 = sorted(int(c[3:]) for c in Pp.columns if c.startswith('xp_'))[:3]
    res = fe2.optimise(Pp, gw3, tm1['squad'], tm1['bank'], tm1['free_transfers'], max_transfers=3)
    b0, _ = fe2.squad_points(Pp, ids, gw3); b1, _ = fe2.squad_points(Pp, res['squad'], gw3)
    xi, cap, vice, bench = fe2.best_xi(ids, Pp, gw3[0])
    P(f"For the test team (FPL team ID 1, \"{tm1['name']}\"): bank £{tm1['bank'] / 10:.1f}m, {tm1['free_transfers']} free transfers. Best GW{gw3[0]} captain: {Pp.name[cap]}, vice {Pp.name[vice]}.")
    moves = ', '.join(f'{Pp.name[o]} out, {Pp.name[i]} in' for o, i in zip(sorted(res["out"], key=lambda q: Pp.pos[q]), sorted(res["inn"], key=lambda q: Pp.pos[q]))) or 'no transfer'
    P(f"Allowing up to 3 transfers over GW{gw3[0]} to GW{gw3[-1]}, the optimiser chooses: <b>{moves}</b>, with {res['hits']} hits, for an expected gain of "
      f"<b>{b1 - b0 - 4 * res['hits']:+.1f}</b> weighted points." + (" It stopped short of 3 transfers because no further move gains more than the 4-point hit (plus the value of saving a transfer)." if len(res['inn']) < 3 else ''))
except Exception as e:
    P(f'(The live team example could not be loaded while building this report: {e}.)')

# ======================================================================= 12 timeline
H1('13. The whole journey: every change, in order')
P('Projects are never a straight line. This is the real order of events, including every failure, so you can tell the story and learn from it.')
TL = [
    ('Idea', 'You wanted to learn PyTorch with a portfolio project. Options: cricket win probability, F1 telemetry, football video tracking, FPL.', 'Chose FPL: free data, a clear weekly test, people care, and it fits tabular and sequence models.'),
    ('Rules', 'Checked the official 2026/27 rules (chips, defensive contributions, bonus changes, deadlines).', 'They became the optimiser\'s constraints.'),
    ('Setup', 'PyTorch and PuLP were not installed.', 'Installed PyTorch 2.15 (Apple GPU available) and PuLP.'),
    ('Data', 'The GitHub archive\'s 2026-27 files stopped at Gameweek 6.', 'Wrote fetch_current.py to pull this season straight from the API.'),
    ('Crash', 'Training crashed with a segmentation fault.', 'PyTorch and scikit-learn each load their own OpenMP threading library on macOS. Fixed with KMP_DUPLICATE_LIB_OK=TRUE and OMP_NUM_THREADS=1.'),
    ('First model', 'Network and gradient boosting both looked far better than form (ranking 0.51, top-10 picks about 10 points).', 'Too good to be true...'),
    ('Solver', 'PuLP 4 changed how variables are created and shipped with no solver.', 'Pinned PuLP 2.9.0, which includes CBC and uses the standard API.'),
    ('THE LEAK', 'A prediction showed Liverpool playing "MUN (H)" six times in one week.', 'Fixture IDs restart every season; joining on fixture ID alone copied rows across seasons (312,497 rows instead of 122,170). Added season to the key. Honest ranking: about 0.25.'),
    ('Illegal squad', 'A hand-picked test squad with 4 players from one club gave nonsense transfers.', 'Optimiser now checks the solver status; app checks the club limit.'),
    ('Pairing', 'Matching sold and bought players crashed when positions did not line up.', 'Pair by position after a valid solve.'),
    ('Notes v1', 'Wrote the first PyTorch study notes PDF.', 'Code examples were run to confirm their outputs.'),
    ('Redesign', 'Rebuilt the app with FPL colours, fonts, shirts, a pitch, cards and charts.', 'The server kept serving the old version; it had to be force-restarted.'),
    ('Drift', 'Season-long totals fell by a third after GW13.', 'Rolling windows were filling with empty future weeks. Held each player\'s form at today\'s level.'),
    ('"Underpredicts"', 'You felt projections were too low.', 'Checked calibration (fine overall), checked a whole past season (bias flipped), added longer windows, per-90, last season, xG team strength and scoring level. Ranking 0.258 to 0.273.'),
    ('Wrong teams', 'Adding team xG produced 21,739 duplicate rows.', 'Players who moved mid-season were linked to their end-of-season club. Took each row\'s team from the fixture. 2022-23 had partial xG, so 0 team xG became missing.'),
    ('Shell leak', 'A test setting (FINAL=0) carried over in the terminal, so the final model was not retrained and the app\'s gradient-boosting model expected different features.', 'Cleared the setting, retrained; the app now reads the feature list stored with each model.'),
    ('Form', 'FPL\'s expected points equalled form. Tested adding more form.', 'It made predictions worse in two seasons. Kept 0% by default, added an optional slider.'),
    ('Planner & differentials', 'Built the Planner (net gain of your own transfers) and Differentials (edge).', 'Tests caught a false "already in your squad" error and buying a player you own.'),
    ('Team strength', 'Goals-based strength followed team IDs across seasons and drifted for future weeks.', 'Switched to team codes and carried values forward.'),
    ('Team form & matchups', 'Added recent results, goal difference, opponent and home/away effects.', 'Neutral in testing; kept and shown in the app.'),
    ('Chances', 'Added goal, assist, 60-minute and clean-sheet models.', 'Clean sheets were under-predicted; rebuilt as a team model.'),
    ('Compare', 'Moved player comparison to its own tab; tied values no longer get a green "best".', ''),
    ('DC & bonus', 'Added defensive-contribution and bonus features and chance models.', 'Old seasons\' DC zeros became missing. A name clash (bonus_rate_r10 also matched the bonus_r pattern) duplicated a column; fixed. Expected bonus made consistent.'),
    ('Layout', 'The player table got crowded.', 'Four fixtures, chances on two short lines, headers that do not wrap.'),
    ('Automation', 'You asked about making changes on users\' real teams.', 'Declined for security and FPL\'s terms; planned a checklist and a late refresh instead.'),
]
T([['Step', 'What happened', 'What we did']] + [[a, b, c] for a, b, c in TL], [2.6, 6.8, 6.6])
H2('The lessons')
B(['<b>When a result looks too good, look for a leak.</b> Check row counts after every join. Look at a few predictions by eye.',
   '<b>Test hunches; do not assume.</b> "Form matters more" sounded right and was wrong.',
   '<b>Always have a baseline.</b> Beating "average of the last 5" is the minimum bar.',
   '<b>Split by time.</b> In forecasting, random splits cheat.',
   '<b>Missing is not zero.</b> Defensive contributions before 2025-26 were unknown, not nothing.',
   '<b>Keep identities stable.</b> Fixture IDs, team IDs and squads all change between seasons.',
   '<b>Negative results are worth reporting.</b> Saying what did not help shows you tested properly.'])

# ======================================================================= 13 limits
H1('14. Limitations')
B(['FPL points are mostly luck in a single match. Even a perfect model of the chances would miss by about 2 points on average.',
   'The model does not read team news beyond FPL\'s injury flags, so late rotation and press-conference news are missed until FPL updates the flag.',
   'Projections further ahead hold form at today\'s level and ignore future transfers, injuries and rotation.',
   'Defensive contributions have only 1.3 seasons of history.',
   'The optimiser plans one set of transfers now; it does not plan a sequence of future transfers or chip timing.',
   'Assist chances are slightly high for top players; clean-sheet chances sit below this season\'s unusually high rate on purpose.',
   'The FPL API and shirt images are unofficial to use and could change without warning.'])

# ======================================================================= 14 running
H1('15. Running it, and where everything lives')
T([['File', 'Purpose'], ['fetch_current.py', 'Download this season from the FPL API.'], ['fpl_data.py', 'Build the table and all 67 features.'], ['fpl_model.py', 'PointsNet, Scaler, predict, load.'],
   ['train_mlp.py', 'Train and test the points models (FEATS= picks a feature set; FINAL=0 skips the final retrain).'], ['train_events.py', 'Train and test the chance models.'],
   ['make_predictions.py', 'Write data/predictions.parquet for every remaining gameweek.'], ['fpl_engine.py', 'Teams, prices, free transfers, best XI, optimiser.'],
   ['app.py, ui.py', 'The Streamlit app and its look.'], ['test_app.py', 'Automated tests of every tab.'],
   ['check_*.py', 'The investigations in this report (calibration, bias, feature sets, form weight, clean sheets, defensive contributions).'],
   ['notes/', 'This report, the study notes, the interview guide, and the scripts that build them.']], [4.2, 11.8])
C('''
cd /Users/pranav_arakkal/Documents/Portfolio/Website/FPL
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
/opt/anaconda3/bin/python3 fetch_current.py        # after each gameweek
/opt/anaconda3/bin/python3 train_mlp.py            # retrain and re-test the points model
/opt/anaconda3/bin/python3 train_events.py         # retrain the chance models
/opt/anaconda3/bin/python3 make_predictions.py     # new predictions
/opt/anaconda3/bin/streamlit run app.py            # the app
/opt/anaconda3/bin/python3 test_app.py             # check every tab
''')

# ======================================================================= 15 exercises
H1('16. Exercises: make it yours')
P('Do these in order. Each changes one thing and you can measure the effect with train_mlp.py (use FINAL=0 to skip the final retrain).')
B(['<b>1. Network size.</b> Try hidden=64 and hidden=256 in PointsNet. Does validation loss improve? (capacity vs overfitting)',
   '<b>2. Dropout.</b> Set drop=0. Watch early stopping trigger earlier. Then 0.4.',
   '<b>3. Learning rate.</b> Try 1e-2 and 1e-4. Too big bounces, too small crawls.',
   '<b>4. One network vs five.</b> Change n=5 to n=1 in fit_nets and compare the test table.',
   '<b>5. DataLoader.</b> Replace the manual batching with TensorDataset and DataLoader(batch_size=512, shuffle=True).',
   '<b>6. Feature importance.</b> Shuffle one feature column in the test set and see how much the error rises. Which features matter most?',
   '<b>7. Player embeddings.</b> Add nn.Embedding(n_players, 8) keyed on the player code and plot the vectors: which players does the network think are alike?',
   '<b>8. Multi-task network.</b> One PyTorch network with several heads: points (MSE), goal and clean sheet (binary cross-entropy). Shared layers learn from all targets.',
   '<b>9. Predict a distribution.</b> Output probabilities for 0-1, 2-3, 4-6 and 7+ points with CrossEntropyLoss. Useful for captaincy risk.',
   '<b>10. Sequence model.</b> Feed each player\'s last 10 matches to an nn.GRU instead of rolling averages.'])

# ======================================================================= 16 glossary and Q&A
H1('17. Glossary')
GL = [('Autograd', 'PyTorch\'s automatic gradient calculation.'), ('Batch', 'A small group of rows used for one training step.'), ('BPS', 'Bonus Points System: the score that decides bonus.'),
      ('Calibration', 'Whether predicted averages match actual averages.'), ('CBIT / CBIRT', 'Clearances, blocks, interceptions, tackles (+ recoveries).'),
      ('Classifier', 'A model that outputs the probability of a yes/no event.'), ('Dropout', 'Randomly zeroing values during training to prevent memorising.'),
      ('Early stopping', 'Stop training when validation loss stops improving; keep the best weights.'), ('Embedding', 'A learned vector of numbers for each category.'),
      ('Ensemble', 'Several models averaged.'), ('Epoch', 'One pass through all training rows.'), ('Feature', 'One input number for a row.'),
      ('Gradient', 'How much the loss changes when a weight changes.'), ('Gradient boosting', 'Many small decision trees, each fixing the last ones\' errors.'),
      ('Leakage', 'Information from the future or the answer sneaking into the inputs.'), ('Loss', 'One number measuring how wrong the predictions are.'),
      ('MILP', 'Mixed-integer linear programming: exact optimisation with yes/no variables.'), ('MSE / RMSE / MAE', 'Squared error / its square root / absolute error.'),
      ('Overfitting', 'Learning the noise in training data so new data is predicted badly.'), ('Poisson', 'A model for counts; P(0) = exp(-mean) gives a clean-sheet chance.'),
      ('ReLU', 'max(0, x): the non-linearity in the network.'), ('Shrinkage', 'Pulling small-sample averages towards the overall average.'),
      ('Spearman correlation', 'Correlation of rankings; our "ranking accuracy".'), ('Standardise', 'Subtract the mean and divide by the standard deviation.'),
      ('Tensor', 'PyTorch\'s array.'), ('Validation set', 'Held-out rows for decisions during training.'), ('Weight decay', 'A pull of weights towards zero.'),
      ('xG / xA', 'Expected goals / expected assists: chance quality.')]
T([['Term', 'Meaning']] + [[a, b] for a, b in GL], [4, 12])
H1('18. Ten questions you should be able to answer')
QA = [('Why split by time?', 'To never learn from the future; it is how the model is used in real life.'),
      ('Why is the ranking accuracy only about 0.26?', 'FPL points are mostly luck in one match. 0.26 against 0.18 for form is a large, consistent edge over a season.'),
      ('Why blend two models?', 'They learn differently, so errors partly cancel; the blend beat both on the test weeks.'),
      ('What did the leak teach you?', 'Check row counts after joins and question results that look too good.'),
      ('Why not give form more weight?', 'Tested it: worse in two seasons. Hot streaks fade.'),
      ('How does the network handle positions?', 'An 8-number learned embedding per position, joined to the features.'),
      ('How do you stop overfitting?', 'Early stopping on a time-based validation set, dropout, weight decay, and an ensemble of 5.'),
      ('Why linear programming for transfers?', 'The rules interact; only an exact search guarantees the best legal answer, and it is fast.'),
      ('How do you know the chances are right?', 'Predicted vs actual rates on unseen weeks, for everyone, starters and the top 200.'),
      ('What would you do next?', 'Predict a distribution, multi-task network, player embeddings, a sequence model, and an automated pre-deadline refresh.')]
T([['Question', 'Short answer']] + [[a, b] for a, b in QA], [5.5, 10.5])

doc = Doc(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title='FPL Points Predictor: the complete guide', author='Pranav Arakkal')
doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
print('built', OUT)
