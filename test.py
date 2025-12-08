"""
Algorithmic Statistical Arbitrage — Triplet-Only Engine (β-neutral, execution-priced, dynamic thresholds)
Rotman International Trading Competition (RITC)

Keeps:
- Three printed tables (Historical Prices, Correlation, Volatility & Beta)
- Average historical divergence table
- Live divergence chart with entry/exit guide lines

Trades:
- Pair-first: trade the max-div stock with the most-opposite div; third leg = HOLD (0) unless neutrality needs it.
- Entry uses execution-priced divergence: SELL@bid divergence, BUY@ask divergence.
- Dynamic thresholds: either "shift" (center on avg divergence) or "widen" (bands widened by |avg_div|).
- β-neutral enforced with execution prices; respects Gross/Net share limits via scaling.
- Exit: dynamic band + net-guard flatten.

Author: GPT (for Yi-Ming Chang)
"""

# %%
import requests
from time import sleep
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import math
import matplotlib.pyplot as plt

# ========= CONFIG =========
API = "http://localhost:9999/v1"
API_KEY = "Rotman"
HDRS = {"X-API-key": API_KEY}

NGN, WHEL, GEAR, RSM1000 = "NGN", "WHEL", "GEAR", "RSM1000"

# Trade/limit config
ORDER_SIZE      = 10_000     # initial target shares PER ACTIVE LEG (before β-neutral & scaling)
MAX_TRADE_SIZE  = 10_000     # per order cap (RITC rule)
GROSS_LIMIT_SH  = 500_000
NET_LIMIT_SH    = 100_000

# Base thresholds in percentage points (pp)
ENTRY_BAND_PCT_BASE = 0.80       # enter if |div| >= 0.80 pp (before dynamic adj)
EXIT_BAND_PCT_BASE  = 0.50       # flatten if |div| < 0.50 pp (before dynamic adj)

# Dynamic modes: "shift" or "widen"
DYN_MODE       = "shift"         # "shift" centers bands on avg divergence; "widen" widens bands by |avg_div|
WIDEN_K_ENTRY  = 0.50            # for "widen": entry_thr = base + k * |avg_div|
WIDEN_K_EXIT   = 0.25            # for "widen": exit_thr  = base + k * |avg_div|

SLEEP_SEC       = 0.5
PRINT_HEARTBEAT = True

# ========= SESSION =========
s = requests.Session()
s.headers.update(HDRS)

# ========= API HELPERS =========
def get_tick_status():
    r = s.get(f"{API}/case"); r.raise_for_status()
    j = r.json()
    return j["tick"], j["status"]

def fetch_all_books():
    """Fetch order books for all tickers once per tick (4 calls total)."""
    tickers = [RSM1000, NGN, WHEL, GEAR]
    out = {}
    for t in tickers:
        r = s.get(f"{API}/securities/book", params={"ticker": t}); r.raise_for_status()
        b = r.json()
        bid = float(b["bids"][0]["price"]) if b.get("bids") else 0.0
        ask = float(b["asks"][0]["price"]) if b.get("asks") else 1e12
        mid = None if (bid == 0.0 and ask == 1e12) else 0.5*(bid+ask)
        out[t] = {"bid": bid, "ask": ask, "mid": mid}
    return out

def positions_map():
    r = s.get(f"{API}/securities"); r.raise_for_status()
    out = {p["ticker"]: int(p.get("position", 0)) for p in r.json()}
    for k in (NGN, WHEL, GEAR, RSM1000):
        out.setdefault(k, 0)
    return out

def place_mkt(ticker: str, action: str, total_qty: int) -> bool:
    """Chunk to MAX_TRADE_SIZE; returns True if all posts succeeded."""
    qty_left, ok_all = int(total_qty), True
    while qty_left > 0:
        q = min(qty_left, MAX_TRADE_SIZE)
        resp = s.post(f"{API}/orders",
                      params={"ticker": ticker, "type":"MARKET", "quantity": int(q), "action": action})
        if PRINT_HEARTBEAT:
            print(f"ORDER {action:<4} {q:6d} {ticker} -> HTTP {resp.status_code} {'OK' if resp.ok else 'FAIL'}")
        if not resp.ok:
            ok_all = False
            break
        qty_left -= q
    return ok_all

# ========= LIMIT HELPERS =========
def within_limits(pos=None):
    p = pos if pos is not None else positions_map()
    gross = abs(p[NGN]) + abs(p[WHEL]) + abs(p[GEAR])
    net   = p[NGN] + p[WHEL] + p[GEAR]
    return (gross <= GROSS_LIMIT_SH) and (abs(net) <= NET_LIMIT_SH)

def scale_triplet_to_limits(actions, qtys, cur_pos):
    """
    actions: dict tkr->'BUY'|'SELL'|'HOLD'
    qtys:    dict tkr->int (>=0)
    cur_pos: dict current positions
    Scales all non-zero legs by the same factor if needed to satisfy Gross/Net share limits.
    """
    p = dict(cur_pos)
    for t in (NGN, WHEL, GEAR):
        a, q = actions[t], qtys[t]
        if a == "BUY":  p[t] = p.get(t,0) + q
        if a == "SELL": p[t] = p.get(t,0) - q

    gross = abs(p[NGN]) + abs(p[WHEL]) + abs(p[GEAR])
    net   = p[NGN] + p[WHEL] + p[GEAR]
    if gross <= GROSS_LIMIT_SH and abs(net) <= NET_LIMIT_SH:
        return qtys

    g_sf = (GROSS_LIMIT_SH / gross) if gross else 1.0
    n_sf = (NET_LIMIT_SH   / abs(net)) if net   else 1.0
    sf   = max(0.0, min(1.0, g_sf, n_sf))

    return {t: int(max(0, math.floor(qtys[t]*sf))) for t in (NGN, WHEL, GEAR)}

# ========= EXIT (FLATTEN) WITH NET GUARD =========
def flatten_full_with_net_guard(tkr: str):
    """
    Fully flatten a single name, clamped by net capacity this tick.
    Prevents breaching NET_LIMIT_SH when unwinding.
    """
    p = positions_map()
    pos = p.get(tkr, 0)
    if pos == 0: return False

    net = p[NGN] + p[WHEL] + p[GEAR]
    L   = NET_LIMIT_SH

    if pos > 0:
        q_cap_net = max(0, L + net)     # long -> SELL q, net' = net - q
        action, qty_target = "SELL", pos
    else:
        q_cap_net = max(0, L - net)     # short -> BUY q, net' = net + q
        action, qty_target = "BUY", -pos

    q = int(min(qty_target, q_cap_net, MAX_TRADE_SIZE))
    if q <= 0:
        if PRINT_HEARTBEAT:
            print(f"[FLAT-BLOCKED] {tkr}: pos={pos:+d}, net={net:+d}, cap={q_cap_net}")
        return False

    if PRINT_HEARTBEAT:
        print(f"[FLAT] {tkr}: {action} {q} (pos={pos:+d}, net={net:+d})")
    return place_mkt(tkr, action, q)

# ========= HISTORICAL / TABLES / BETAS =========
def load_historical():
    r = s.get(f"{API}/news"); r.raise_for_status()
    news = r.json()
    if not news:
        print("No news yet. Start the case and ensure table is published.")
        return None
    soup = BeautifulSoup(news[0].get("body",""), "html.parser")
    table = soup.find("table")
    if not table:
        print("No <table> in news body.")
        return None

    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == 5:
            rows.append(cols)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["Tick"] = df["Tick"].astype(int)
    for c in ["RSM1000", "NGN", "WHEL", "GEAR"]:
        df[c] = df[c].astype(float)
    return df

def print_three_tables_and_betas(df_hist):
    # 1) Historical price table
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
    print("\nHistorical Price Data:\n")
    print(df_hist.to_string(index=False))

    # 2) Correlation on tick returns
    returns = df_hist[["RSM1000", "NGN", "WHEL", "GEAR"]].pct_change().dropna()
    corr = returns.corr()
    print("\nHistorical Correlation:\n")
    print(corr.to_string())

    # 3) Volatility & beta (vs RSM1000)
    tick_vol = returns.std()
    idx_var  = returns["RSM1000"].var()
    if idx_var == 0 or np.isnan(idx_var): idx_var = 1e-12
    beta_map = {t: float(np.cov(returns[t], returns["RSM1000"])[0,1] / idx_var)
                for t in ["RSM1000","NGN","WHEL","GEAR"]}
    vol_beta_df = pd.DataFrame({
        "Tick Volatility": tick_vol,
        "Beta vs RSM1000": [beta_map[t] for t in tick_vol.index]
    })
    print("\nHistorical Volatility and Beta:\n")
    print(vol_beta_df.to_string())

    # 4) Average divergence (based on historical PTD vs β·index)
    base = {t: df_hist[t].iloc[0] for t in ["RSM1000", "NGN", "WHEL", "GEAR"]}
    ptd = {t: (df_hist[t] / base[t]) - 1.0 for t in base}
    div = {
        "NGN":  (ptd["NGN"]  - beta_map["NGN"]  * ptd["RSM1000"]) * 100.0,
        "WHEL": (ptd["WHEL"] - beta_map["WHEL"] * ptd["RSM1000"]) * 100.0,
        "GEAR": (ptd["GEAR"] - beta_map["GEAR"] * ptd["RSM1000"]) * 100.0,
    }
    avg_div = {k: float(np.mean(v)) for k, v in div.items()}
    avg_div_df = pd.DataFrame.from_dict(avg_div, orient="index", columns=["Avg Divergence (pp)"])
    print("\nAverage Historical Divergence (percentage points):\n")
    print(avg_div_df.to_string())

    # return stock betas + avg divergences
    betas_only = {k: v for k, v in beta_map.items() if k != "RSM1000"}
    return betas_only, avg_div

# ========= ACTION DECISION / EXECUTION PRICES =========
def exec_px(t, action, quotes):
    if action == "SELL": return quotes[t]["bid"]
    if action == "BUY":  return quotes[t]["ask"]
    return 0.0

def signed(action):  # +1 for SELL, -1 for BUY, 0 for HOLD
    return +1 if action == "SELL" else (-1 if action == "BUY" else 0)

def per_name_thresholds(avg_div):
    """Return entry/exit thresholds per ticker depending on DYN_MODE."""
    if DYN_MODE == "widen":
        thr_entry = {t: ENTRY_BAND_PCT_BASE + WIDEN_K_ENTRY * abs(avg_div.get(t, 0.0)) for t in (NGN,WHEL,GEAR)}
        thr_exit  = {t: EXIT_BAND_PCT_BASE  + WIDEN_K_EXIT  * abs(avg_div.get(t, 0.0)) for t in (NGN,WHEL,GEAR)}
    else:
        # "shift": same global magnitudes; bands applied to (div - avg_div)
        thr_entry = {t: ENTRY_BAND_PCT_BASE for t in (NGN,WHEL,GEAR)}
        thr_exit  = {t: EXIT_BAND_PCT_BASE  for t in (NGN,WHEL,GEAR)}
    return thr_entry, thr_exit

def best_pair_actions(div_if_sell, div_if_buy, avg_div, thr_entry):
    """
    Build actions by:
      1) For each stock, pick the action with the larger entry 'score' (>= threshold),
         where score = +div_if_sell for SELL; score = +(-div_if_buy) for BUY (absolute amount beyond zero).
         In 'shift' mode, compare to (div - avg).
      2) Choose the max-score stock; pair it with the strongest opposite-sign stock.
      3) Third leg = HOLD.
    Returns actions dict: {t: 'SELL'|'BUY'|'HOLD'}
    """
    candid = {}
    for t in (NGN, WHEL, GEAR):
        if DYN_MODE == "shift":
            sell_metric = div_if_sell[t] - avg_div.get(t,0.0)
            buy_metric  = div_if_buy[t]  - avg_div.get(t,0.0)
        else:
            sell_metric = div_if_sell[t]
            buy_metric  = div_if_buy[t]
        # entry checks
        act = "HOLD"; score = 0.0
        if sell_metric >= thr_entry[t]:
            act, score = "SELL", sell_metric
        if (-buy_metric) >= thr_entry[t] and (-buy_metric) > score:  # buy_metric negative is favorable
            act, score = "BUY", (-buy_metric)
        candid[t] = (act, score)

    # choose primary with largest score
    t_primary = max((t for t in candid if candid[t][0] != "HOLD"), key=lambda x: candid[x][1], default=None)
    if t_primary is None:
        return {NGN:"HOLD", WHEL:"HOLD", GEAR:"HOLD"}

    act_primary = candid[t_primary][0]
    # find most-opposite
    opp_needed = "BUY" if act_primary == "SELL" else "SELL"
    others = [t for t in (NGN,WHEL,GEAR) if t != t_primary and candid[t][0] == opp_needed]
    if others:
        t_pair = max(others, key=lambda t: candid[t][1])
        actions = {t:"HOLD" for t in (NGN,WHEL,GEAR)}
        actions[t_primary] = act_primary
        actions[t_pair] = opp_needed
        return actions
    else:
        # no opposite available -> single
        actions = {t:"HOLD" for t in (NGN,WHEL,GEAR)}
        actions[t_primary] = act_primary
        return actions

# ========= β-NEUTRAL TRIPLET SOLVER (uses execution prices) =========
def beta_neutral_triplet(actions, quotes, betas, order_size=ORDER_SIZE, max_iter=3):
    """
    Enforce sum(s_i * q_i * px_i * beta_i) = 0 using execution prices:
      SELL -> px = bid; BUY -> px = ask; HOLD -> q = 0
    """
    q = {t: (order_size if actions[t] in ("BUY","SELL") else 0) for t in (NGN,WHEL,GEAR)}

    for _ in range(max_iter):
        active = [t for t in (NGN,WHEL,GEAR) if actions[t] in ("BUY","SELL") and q[t] > 0]
        if len(active) <= 1:
            return q  # nothing to neutralize

        contrib = {t: signed(actions[t]) * q[t] * exec_px(t, actions[t], quotes) * betas.get(t,0.0)
                   for t in active}
        total = sum(contrib.values())
        if abs(total) < 1e-9:
            return q

        t_adj = max(active, key=lambda t: abs(contrib[t]))
        s = signed(actions[t_adj]); p = exec_px(t_adj, actions[t_adj], quotes); b = betas.get(t_adj, 0.0)
        denom = s * p * b
        if abs(denom) < 1e-12:
            q[t_adj] = 0  # can't adjust, drop it
            continue

        q_new_f = q[t_adj] - (total / denom)
        q_new = int(min(MAX_TRADE_SIZE, max(0, math.floor(q_new_f))))
        if q_new <= 0:
            q[t_adj] = 0
        else:
            q[t_adj] = q_new

    return q

# ========= MAIN =========
def main():
    # 1) Historical / Three tables + betas + avg divergence
    df_hist = load_historical()
    if df_hist is None: return
    betas, avg_div = print_three_tables_and_betas(df_hist)
    thr_entry, thr_exit = per_name_thresholds(avg_div)

    # 2) PTD bases (first mids seen)
    base = {t: None for t in (RSM1000, NGN, WHEL, GEAR)}

    # 3) Live plotting (kept)
    ticks, d_ngn, d_whe, d_ger = [], [], [], []
    plt.ion()
    fig, ax = plt.subplots()
    l1, = ax.plot([], [], label="NGN")
    l2, = ax.plot([], [], label="WHEL")
    l3, = ax.plot([], [], label="GEAR")
    ax.set_title("Divergence (pp) vs β·Index (mid-based)")
    ax.set_xlabel("Tick"); ax.set_ylabel("Divergence (pp)")
    ax.axhline( ENTRY_BAND_PCT_BASE,  color='red',    linestyle='--', linewidth=1,   label='+Entry (base)')
    ax.axhline(-ENTRY_BAND_PCT_BASE, color='green',  linestyle='--', linewidth=1,   label='-Entry (base)')
    ax.axhline( EXIT_BAND_PCT_BASE,   color='orange', linestyle='--', linewidth=0.8, label='+Exit (base)')
    ax.axhline(-EXIT_BAND_PCT_BASE,  color='orange', linestyle='--', linewidth=0.8, label='-Exit (base)')
    ax.grid(True); ax.legend()

    # 4) Live loop
    tick, status = get_tick_status()
    while status == "ACTIVE":
        quotes = fetch_all_books()
        if any(quotes[t]["mid"] is None for t in (RSM1000, NGN, WHEL, GEAR)):
            tick, status = get_tick_status(); continue

        # set bases lazily (using mid)
        for t in base:
            if base[t] is None:
                base[t] = quotes[t]["mid"]

        # Index PTD uses mid (we don't trade index)
        ptd_idx = (quotes[RSM1000]["mid"]/base[RSM1000]) - 1.0

        # Divergences for plotting (mid-based)
        div_mid = {}
        for t in (NGN, WHEL, GEAR):
            ptd_mid = (quotes[t]["mid"]/base[t]) - 1.0
            div_mid[t] = (ptd_mid - betas[t] * ptd_idx) * 100.0

        # Live chart update
        ticks.append(tick); d_ngn.append(div_mid[NGN]); d_whe.append(div_mid[WHEL]); d_ger.append(div_mid[GEAR])
        l1.set_data(ticks, d_ngn); l2.set_data(ticks, d_whe); l3.set_data(ticks, d_ger)
        ax.relim(); ax.autoscale_view(); plt.pause(0.01)

        # Execution-priced divergences for entry
        div_if_sell, div_if_buy = {}, {}
        for t in (NGN, WHEL, GEAR):
            ptd_bid = (quotes[t]["bid"]/base[t]) - 1.0
            ptd_ask = (quotes[t]["ask"]/base[t]) - 1.0
            div_if_sell[t] = (ptd_bid - betas[t] * ptd_idx) * 100.0   # SELL uses bid
            div_if_buy[t]  = (ptd_ask - betas[t] * ptd_idx) * 100.0   # BUY  uses ask

        # ===== EXIT FIRST: dynamic-band flatten (mid-based) =====
        cur_pos = positions_map()
        for t in (NGN, WHEL, GEAR):
            # dynamic exit test
            if DYN_MODE == "shift":
                exit_metric = abs(div_mid[t] - avg_div.get(t,0.0))
                thr_ex = thr_exit[t]
            else:  # widen
                exit_metric = abs(div_mid[t])
                thr_ex = thr_exit[t]
            if cur_pos.get(t,0) != 0 and exit_metric < thr_ex:
                flatten_full_with_net_guard(t)

        # ===== ENTRY: max-with-most-opposite pairing (execution-priced) =====
        actions = best_pair_actions(div_if_sell, div_if_buy, avg_div, thr_entry)

        if all(a == "HOLD" for a in actions.values()):
            if PRINT_HEARTBEAT:
                mx = max((abs(div_mid[t]) for t in (NGN,WHEL,GEAR)), default=0.0)
                print(f"[Tick {tick}] No entry (dynamic). max |div_mid|={mx:.3f}")
            tick, status = get_tick_status(); continue

        # β-neutral quantities (execution prices)
        qty_beta = beta_neutral_triplet(actions, quotes, betas, order_size=ORDER_SIZE)

        if sum(qty_beta.values()) == 0:
            if PRINT_HEARTBEAT:
                print(f"[Tick {tick}] β-neutral eliminated trade (all zero).")
            tick, status = get_tick_status(); continue

        # Scale to share limits (Gross/Net)
        cur_pos = positions_map()
        qty_exec = scale_triplet_to_limits(actions, qty_beta, cur_pos)

        if sum(qty_exec.values()) == 0:
            if PRINT_HEARTBEAT:
                print(f"[Tick {tick}] Scaled to zero by limits.")
            tick, status = get_tick_status(); continue

        # Execute (only non-zero legs)
        if PRINT_HEARTBEAT:
            msg = " | ".join([f"{t}:{actions[t]} {qty_exec[t]}" for t in (NGN,WHEL,GEAR) if qty_exec[t] > 0])
            print(f"\n[EXEC T{tick}] {msg}")
        legs_sent = 0
        for t in (NGN, WHEL, GEAR):
            a, q = actions[t], qty_exec[t]
            if a in ("BUY","SELL") and q > 0:
                ok = place_mkt(t, a, q)
                if ok: legs_sent += 1

        # Sleep only after a PAIR (2+ legs) to avoid over-trading on singles
        if legs_sent >= 2:
            sleep(SLEEP_SEC)

        tick, status = get_tick_status()

    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()
# %%
