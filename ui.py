"""Look and feel: CSS, colours, shirt images and the HTML pieces (pitch, player table, fixture ticker, cards)."""
import html
import numpy as np, pandas as pd

PURPLE, DEEP, GREEN, CYAN, PINK, MUTED = '#37003c', '#1b0021', '#00ff87', '#04f5ff', '#e90052', '#b9a7c2'
POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POS_COL = {1: '#ebff00', 2: '#00ff87', 3: '#04f5ff', 4: '#e90052'}
FDR = {0: ('#3a2342', '#b9a7c2'), 1: ('#257d5a', '#ffffff'), 2: ('#00ff87', '#1b0021'), 3: ('#e7e7e7', '#1b0021'), 4: ('#ff1751', '#ffffff'), 5: ('#861134', '#ffffff')}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp {background: radial-gradient(1200px 600px at 85% -10%, #4a0a55 0%, rgba(27,0,33,0) 60%), linear-gradient(180deg, #24002c 0%, #1b0021 40%);}
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp li, [data-testid="stMarkdownContainer"] div {font-family: 'Inter', sans-serif;}
h1, h2, h3, h4, .hero-title, .sec, .card .v, .pname, .bigname {font-family: 'Outfit', sans-serif !important;}
.block-container {padding-top: 1.4rem; max-width: 1280px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #37003c 0%, #22002a 100%); border-right: 1px solid rgba(255,255,255,.06);}
[data-testid="stHeader"] {background: transparent;}
/* hide Streamlit Cloud's Share / Star / GitHub / Deploy buttons, the main menu and the hosting badges. The sidebar toggle stays. */
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stMainMenu"], #MainMenu,
[data-testid="stToolbarActionButton"], [data-testid="stStatusWidget"] a, [class*="viewerBadge"], [data-testid="appCreatorAvatar"],
a[href*="streamlit.io/cloud"], a[href^="https://github.com"][target="_blank"][class*="st-"] {display: none !important;}
footer {visibility: hidden;}
.credit {margin-top: 34px; padding: 16px 0 6px 0; border-top: 1px solid rgba(255,255,255,.08); color: #b9a7c2; font-size: .82rem;
  display: flex; flex-wrap: wrap; gap: 6px 18px; justify-content: space-between; align-items: center;}
.credit b {color: #ffffff; font-family: 'Outfit';} .credit a {color: #00ff87; text-decoration: none;} .credit a:hover {text-decoration: underline;}
.sb-credit {margin-top: 18px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.1); color: #b9a7c2; font-size: .8rem; line-height: 1.5;}
.sb-credit b {color: #ffffff;} .sb-credit a {color: #00ff87; text-decoration: none;}
.hero {border-radius: 18px; padding: 22px 26px; margin-bottom: 14px; position: relative; overflow: hidden;
  background: linear-gradient(110deg, #37003c 0%, #5a0a63 45%, #0b6a5a 100%); border: 1px solid rgba(255,255,255,.08);}
.hero:after {content: ''; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,255,135,.35), rgba(0,255,135,0) 70%);}
.hero-title {font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px; line-height: 1.1;}
.hero-title span {color: #00ff87;}
.hero-sub {color: #e6d9ec; margin-top: 6px; font-size: .95rem;}
.pill {display: inline-block; padding: 4px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px; margin-top: 10px;
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.15);}
.pill.g {background: #00ff87; color: #1b0021; border: none;}
.sec {font-size: 1.25rem; font-weight: 700; margin: 18px 0 8px 0;}
.sec small {font-family: 'Inter'; font-weight: 400; color: #b9a7c2; font-size: .85rem; margin-left: 8px;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; margin: 6px 0 4px 0;}
.card {background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 12px 14px;}
.card .v {font-size: 1.6rem; font-weight: 700; line-height: 1.15;} .card .v.g {color: #00ff87;} .card .v.c {color: #04f5ff;}
.card .l {color: #b9a7c2; font-size: .78rem; margin-top: 2px;}
.pitch {border-radius: 18px; padding: 18px 8px 10px 8px; position: relative; border: 1px solid rgba(255,255,255,.12);
  background: repeating-linear-gradient(180deg, #0f8a4a 0px, #0f8a4a 46px, #0c7d42 46px, #0c7d42 92px);}
.pitch:before {content: ''; position: absolute; inset: 10px; border: 2px solid rgba(255,255,255,.35); border-radius: 6px; pointer-events: none;}
.pitch:after {content: ''; position: absolute; left: 50%; top: 10px; width: 150px; height: 60px; transform: translateX(-50%);
  border: 2px solid rgba(255,255,255,.35); border-top: none; pointer-events: none;}
.prow {display: flex; justify-content: center; gap: 2.2%; margin: 10px 0 16px 0; position: relative; z-index: 1;}
.pl {width: 104px; text-align: center; position: relative;}
.pl img {width: 62px; height: 62px; object-fit: contain; filter: drop-shadow(0 4px 6px rgba(0,0,0,.35));}
.pname {background: #ffffff; color: #37003c; font-weight: 700; font-size: .8rem; border-radius: 6px 6px 0 0; padding: 2px 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.pxp {background: #37003c; color: #00ff87; font-weight: 700; font-size: .8rem; border-radius: 0 0 6px 6px; padding: 2px 4px;}
.pfix {font-size: .68rem; color: #ffffff; background: rgba(0,0,0,.35); border-radius: 4px; margin-top: 3px; padding: 1px 3px;}
.badge {position: absolute; top: -2px; right: 14px; width: 22px; height: 22px; border-radius: 50%; font-weight: 800; font-size: .75rem;
  line-height: 22px; background: #1b0021; color: #00ff87; border: 2px solid #00ff87; font-family: 'Outfit';}
.badge.v {color: #04f5ff; border-color: #04f5ff;}
.flag {position: absolute; top: -2px; left: 14px; width: 20px; height: 20px; border-radius: 50%; background: #ffab1b; color: #1b0021;
  font-weight: 800; font-size: .75rem; line-height: 20px;}
.flag.r {background: #e90052; color: #fff;}
.bench {background: rgba(0,0,0,.25); border-radius: 0 0 18px 18px; margin: -6px 0 0 0; padding: 12px 8px 6px 8px; border: 1px solid rgba(255,255,255,.08); border-top: none;}
.bench .lab {text-align: center; color: #b9a7c2; font-size: .75rem; letter-spacing: 1px; margin-bottom: 4px;}
table.fpl, table.tick {width: 100%; border-collapse: separate !important; border: none !important; margin: 0;}
table.fpl {border-spacing: 0 5px; font-size: .86rem;}
table.fpl th, table.fpl td, table.tick th, table.tick td {border: none !important;}
[data-testid="stMarkdownContainer"] table {border: none !important;}
table.fpl th {white-space: nowrap; text-align: left; color: #b9a7c2; font-weight: 500; font-size: .74rem; text-transform: uppercase; letter-spacing: .6px; padding: 4px 8px;}
table.fpl td {background: rgba(255,255,255,.04); padding: 7px 8px; vertical-align: middle;}
table.fpl tr td:first-child {border-radius: 10px 0 0 10px;} table.fpl tr td:last-child {border-radius: 0 10px 10px 0;}
table.fpl tr:hover td {background: rgba(0,255,135,.07);}
.who {display: flex; align-items: center; gap: 10px;} .who img {width: 34px; height: 34px; object-fit: contain;}
.who b {font-family: 'Outfit'; font-size: .95rem;} .who .t {color: #b9a7c2; font-size: .75rem;}
.pos {display: inline-block; font-size: .68rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; color: #1b0021; margin-left: 4px;}
.fxs {white-space: nowrap;} .fx {display: inline-block; min-width: 46px; text-align: center; font-size: .7rem; font-weight: 600; padding: 3px 4px; border-radius: 5px; margin: 1px;}
.xpbar {position: relative; height: 22px; background: rgba(255,255,255,.06); border-radius: 6px; min-width: 110px; overflow: hidden;}
.xpbar i {position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, #04f5ff, #00ff87); border-radius: 6px;}
.xpbar span {position: relative; font-weight: 800; color: #1b0021; padding-left: 8px; line-height: 22px; font-family: 'Outfit';}
.tick {border-spacing: 3px; font-size: .72rem; table-layout: fixed;}
.tick td {text-align: center; border-radius: 6px; padding: 6px 1px; font-weight: 600; white-space: nowrap; overflow: hidden;}
.tick td.tm {text-align: left; background: transparent; font-family: 'Outfit'; font-weight: 700; width: 24%; text-overflow: ellipsis;}
.tick th {color: #b9a7c2; font-weight: 500; font-size: .72rem;}
.move {display: grid; grid-template-columns: 1fr 50px 1fr; align-items: center; gap: 8px; margin: 8px 0;}
.mv {display: flex; align-items: center; gap: 12px; border-radius: 14px; padding: 10px 14px; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);}
.mv.out {border-left: 4px solid #e90052;} .mv.in {border-left: 4px solid #00ff87;}
.mv img {width: 44px; height: 44px; object-fit: contain;} .mv .bigname {font-weight: 700; font-size: 1.05rem;} .mv .m {color: #b9a7c2; font-size: .8rem;}
.arrow {text-align: center; font-size: 1.6rem; color: #00ff87; font-weight: 800;}
.gain {display: inline-block; background: #00ff87; color: #1b0021; font-weight: 800; font-family: 'Outfit'; border-radius: 10px; padding: 6px 14px; font-size: 1.1rem;}
.chn {font-size: .72rem; color: #e6d9ec; white-space: nowrap;} .chn b {color: #04f5ff; font-weight: 700;}
.cmp td.best {color: #00ff87; font-weight: 800;}
.note {color: #b9a7c2; font-size: .84rem; line-height: 1.5;}
div[data-testid="stSegmentedControl"] button, div[data-testid="stPills"] button {font-family: 'Outfit' !important; font-weight: 600;}
</style>
"""


SITE = 'https://pranav-analysis.arakkal-pranav.workers.dev'
LINKEDIN = 'https://www.linkedin.com/in/pranav-arakkal-323691249'          # your LinkedIn profile, e.g. https://www.linkedin.com/in/your-name  (the link is hidden while this is blank)


def _links(sep=' · '):
    out = [f'<a href="{SITE}" target="_blank" rel="noopener">Portfolio ↗</a>']
    if LINKEDIN: out.append(f'<a href="{LINKEDIN}" target="_blank" rel="noopener">LinkedIn ↗</a>')
    out.append('<a href="mailto:arakkal.pranav@gmail.com">Email</a>')
    return sep.join(out)


CREDIT = (f'<div class="credit"><span>Built by <b>Pranav Arakkal</b> · data, models and app</span><span>{_links()}</span>'
          '<span>Not affiliated with the Premier League or Fantasy Premier League. Data: FPL API.</span></div>')
SB_CREDIT = f'<div class="sb-credit">Built by <b>Pranav Arakkal</b><br>{_links()}</div>'


def shirt(team_code, gk=False):
    return f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{int(team_code)}{'_1' if gk else ''}-110.webp"


def esc(s): return html.escape(str(s))


def cards(items):
    """items: list of (value, label, colour class '' / 'g' / 'c')."""
    return '<div class="cards">' + ''.join(f'<div class="card"><div class="v {c}">{v}</div><div class="l">{l}</div></div>' for v, l, c in items) + '</div>'


def fx_chip(opp, fdr):
    bg, fg = FDR.get(int(fdr), FDR[3])
    return f'<span class="fx" style="background:{bg};color:{fg}">{esc(opp) if opp != "-" else "BLANK"}</span>'


def _player(P, i, val, fix, cap=None, vice=None):
    r = P.loc[i]
    b = '<div class="badge">C</div>' if i == cap else ('<div class="badge v">V</div>' if i == vice else '')
    ch = r.chance
    f = '' if pd.isna(ch) or ch >= 100 else f'<div class="flag {"r" if ch <= 25 else ""}">!</div>'
    g = fix_gw.get('gw')
    tip = ''
    if g and f'p_goal_{g}' in r.index:
        tip = (f'Goal {100 * r[f"p_goal_{g}"]:.0f}% · Assist {100 * r[f"p_assist_{g}"]:.0f}% · ' if r.pos != 1 else '') + \
              (f'Clean sheet {100 * r[f"p_cs_{g}"]:.0f}% · ' if r.pos != 4 else '') + \
              (f'Def. contribution {100 * r[f"p_dc_{g}"]:.0f}% · ' if r.pos != 1 else '') + f'Bonus {100 * r[f"p_bonus_{g}"]:.0f}% · 60+ mins {100 * r[f"p_60_{g}"]:.0f}%'
    return (f'<div class="pl" title="{esc(tip)}">{b}{f}<img src="{shirt(r.team_code, r.pos == 1)}"><div class="pname">{esc(r["name"])}</div>'
            f'<div class="pxp">{val:.1f} pts</div><div class="pfix">{esc(fix)}</div></div>')


fix_gw = {}


def pitch(P, xi, bench, cap, vice, value, fix, gw=None):
    """value, fix: dicts id -> number / text shown under each shirt. Hovering a shirt shows his chances for gameweek gw."""
    fix_gw['gw'] = gw
    rows = ''
    for p in (1, 2, 3, 4):
        ids = [i for i in xi if P.pos[i] == p]
        rows += '<div class="prow">' + ''.join(_player(P, i, value[i], fix[i], cap, vice) for i in ids) + '</div>'
    b = ''.join(_player(P, i, value[i], fix[i]) for i in bench)
    return f'<div class="pitch">{rows}</div><div class="bench"><div class="lab">BENCH</div><div class="prow" style="margin:4px 0 6px 0">{b}</div></div>'


def chances(r, g):
    """Compact 'G 56% · A 12% · CS 26%' for one gameweek. Clean sheets only matter for GK, DEF and MID."""
    if f'p_goal_{g}' not in r.index: return ''
    attack = [] if r.pos == 1 else [f'<b>G</b> {100 * r[f"p_goal_{g}"]:.0f}%', f'<b>A</b> {100 * r[f"p_assist_{g}"]:.0f}%']
    if f'p_bonus_{g}' in r.index: attack.append(f'<b>B</b> {100 * r[f"p_bonus_{g}"]:.0f}%')
    defend = []
    if r.pos != 4: defend.append(f'<b>CS</b> {100 * r[f"p_cs_{g}"]:.0f}%')
    if r.pos in (2, 3) and f'p_dc_{g}' in r.index: defend.append(f'<b>DC</b> {100 * r[f"p_dc_{g}"]:.0f}%')
    return '<div class="chn">' + ' · '.join(attack) + '</div>' + ('<div class="chn">' + ' · '.join(defend) + '</div>' if defend else '')


def player_table(P, ids, gws, value_col, max_val=None, extra=None, show_fix=4):
    """HTML player list with shirts, fixture chips, next-gameweek chances and an expected-points bar."""
    max_val = max_val or max(P.loc[ids, value_col].max(), 1)
    head = (f'<tr><th>Player</th><th>Price</th><th>Owned</th><th>Form</th><th>Next fixtures</th><th>GW{gws[0]} chances</th><th>Projected</th><th>Pts / £m</th>'
            + (f'<th>{extra[0]}</th>' if extra else '') + '</tr>')
    body = ''
    for i in ids:
        r = P.loc[i]
        fx = ''.join(fx_chip(r[f'opp_{g}'].split(',')[0].strip() + (' +1' if ',' in r[f'opp_{g}'] else ''), r[f'fdr_{g}']) for g in gws[:show_fix])
        w = max(0, min(100, 100 * r[value_col] / max_val))
        news = f'<div class="t" style="color:#ffab1b">{esc(r.news)}</div>' if isinstance(r.news, str) and r.news else ''
        body += (f'<tr><td><div class="who"><img src="{shirt(r.team_code, r.pos == 1)}"><div><b>{esc(r["name"])}</b>'
                 f'<span class="pos" style="background:{POS_COL[r.pos]}">{POS[r.pos]}</span><div class="t">{esc(r.team)}</div>{news}</div></div></td>'
                 f'<td>£{r.price:.1f}m</td><td>{r.selected:.1f}%</td><td>{r.form:.1f}</td><td class="fxs">{fx}</td><td>{chances(r, gws[0])}</td>'
                 f'<td><div class="xpbar"><i style="width:{w:.0f}%"></i><span>{r[value_col]:.1f}</span></div></td><td>{r[value_col] / r.price:.2f}</td>'
                 + (f'<td>{extra[1](i)}</td>' if extra else '') + '</tr>')
    return f'<table class="fpl">{head}{body}</table>'


def ticker(rows, gws):
    """rows: list of (label, [(opp, fdr) per gw]). Fixture difficulty grid."""
    head = '<tr><th></th>' + ''.join(f'<th>GW{g}</th>' for g in gws) + '</tr>'
    body = ''
    for lab, cells in rows:
        tds = ''
        for opp, f in cells:
            bg, fg = FDR.get(int(f), FDR[3]); tds += f'<td style="background:{bg};color:{fg}">{esc(opp) if opp != "-" else "BLANK"}</td>'
        body += f'<tr><td class="tm">{lab}</td>{tds}</tr>'
    return f'<table class="tick">{head}{body}</table>'


def plot_style(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Outfit, sans-serif', color='#ffffff', size=13), legend=dict(orientation='h', y=1.12, x=0),
                      xaxis=dict(gridcolor='rgba(255,255,255,.06)', zeroline=False), yaxis=dict(gridcolor='rgba(255,255,255,.08)', zeroline=False),
                      hoverlabel=dict(bgcolor='#37003c', font_family='Inter'))
    return fig
