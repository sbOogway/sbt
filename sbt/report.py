import webbrowser
from pathlib import Path

import pandas as pd
from nautilus_trader.analysis import create_tearsheet
from nautilus_trader.backtest.engine import BacktestEngine


def _positions_json(positions: pd.DataFrame) -> str:
    rows = []
    for row in positions.itertuples():
        ts = int(pd.Timestamp(row.ts_opened).timestamp())
        is_long = getattr(row, "is_long", None)
        if is_long is None:
            side = getattr(row, "entry", "")
            is_long = side == "BUY"
        avg_px = float(getattr(row, "avg_px_open", 0))
        rows.append(
            f'{{"time":{ts},"price":{avg_px},"side":"{"BUY" if is_long else "SELL"}"}}'
        )
    return "[" + ",".join(rows) + "]"


_TV_EXCHANGE_PREFIX = {
    "binance": "BINANCE",
    "binanceus": "BINANCEUS",
    "bybit": "BYBIT",
    "okx": "OKX",
    "kucoin": "KUCOIN",
    "bitget": "BITGET",
}

_TV_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "8h": "480",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


def _chart_block(
    positions_json: str, pair: str, exchange: str = "", interval: str = "5m"
) -> str:
    prefix = _TV_EXCHANGE_PREFIX.get((exchange or "").lower())
    if prefix is None:
        return (
            "<p>TradingView chart unavailable: no symbol mapping for exchange "
            f"'{exchange or 'unknown'}'.</p>"
        )
    tv_symbol = f"{prefix}:{pair.replace('/', '')}"
    tv_interval = _TV_INTERVALS.get(interval, "60")
    return f"""
<style>
#tv-wrap {{ width:100%;height:90vh;margin:0;padding:0 }}
</style>
<div id="tv-wrap">
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div id="tv-chart" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  var positions = {positions_json};
  var symbolName = "{tv_symbol}";
  new TradingView.widget({{
    container_id: "tv-chart",
    symbol: symbolName,
    interval: "{tv_interval}",
    timezone: "UTC",
    theme: "dark",
    style: "1",
    locale: "en",
    hide_top_toolbar: false,
    hide_legend: false,
    allow_symbol_change: true,
    save_image: false,
    enable_publishing: false,
    hideideas: true,
    autosize: true,
    toolbar_bg: "#1a1a2e",
    studies: [],
    disabled_features: ["header_widget", "header_symbol_search"],
    enabled_features: [],
  }});
  window.addEventListener('load', function() {{
    function drawPositions(widget) {{
      if (!widget.chart) return;
      var chart = widget.chart();
      if (!chart) return;
      positions.forEach(function(p) {{
        var shapeOpts = {{
          text: p.side === 'BUY' ? 'L' : 'S',
          shape: p.side === 'BUY' ? 'arrow_up' : 'arrow_down',
          overrides: {{
            color: p.side === 'BUY' ? '#26a69a' : '#ef5350',
            textcolor: '#fff',
            fontsize: 10,
            bold: true,
          }},
        }};
        try {{
          chart.createShape({{ time: p.time, price: p.price }}, shapeOpts);
        }} catch(e) {{}}
      }});
    }}
    var attempts = 0;
    var id = setInterval(function() {{
      attempts++;
      if (window.TradingView && window.TradingView.widget && window.TradingView.widget.instances) {{
        var instances = window.TradingView.widget.instances;
        if (instances.length > 0) {{
          var w = instances[instances.length - 1];
          if (w && w.chart) {{
            clearInterval(id);
            w.onChartReady(function() {{ drawPositions(w); }});
          }}
        }}
      }}
      if (attempts > 50) clearInterval(id);
    }}, 100);
  }});
  </script>
</div>
</div>
"""


def _df_to_html_table(df: pd.DataFrame) -> str:
    if not len(df):
        return "<p>None</p>"
    return df.to_html(
        classes="report-table sortable",
        border=0,
        escape=True,
        sparsify=False,
    )


def _fmt_single_stat(key: str, value) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    kl = key.lower()
    if "win rate" in kl:
        return f"{value * 100:.2f}%"
    if "sharpe" in kl or "calmar" in kl or "sortino" in kl:
        return f"{value:.2f}"
    if "annualized return" in kl:
        return f"{value * 100:.2f}%"
    if "max drawdown" in kl:
        return f"{value * 100:.2f}%"
    if "returns volatility" in kl:
        return f"{value * 100:.2f}%"
    if "average" in kl and "return" in kl:
        return f"{value * 100:.2f}%"
    if any(x in kl for x in ("winner", "loser", "expectancy")):
        return f"${value:+,.2f}"
    if "profit factor" in kl or "risk return" in kl:
        return f"{value:.2f}"
    return str(value)


def _process_stats(
    stats_pnls: dict,
    stats_returns: dict,
    stats_general: dict,
) -> str:
    rows: list[tuple[str, str]] = []

    pnl = stats_pnls.get("PnL (total)")
    pnl_pct = stats_pnls.get("PnL% (total)")
    if pnl is not None and isinstance(pnl, (int, float)):
        if pnl_pct is not None and isinstance(pnl_pct, (int, float)):
            rows.append(("PnL", f"${pnl:+,.2f} ({pnl_pct:+.2f}%)"))
        else:
            rows.append(("PnL", f"${pnl:+,.2f}"))

    combined = {**stats_pnls, **stats_returns, **stats_general}
    for skip in ("PnL (total)", "PnL% (total)"):
        combined.pop(skip, None)

    for key, value in combined.items():
        rows.append((key, _fmt_single_stat(key, value)))

    def _color(val: str) -> str:
        s = val.lstrip()
        if s.startswith("+") or s.startswith("$+"):
            return f'<span class="positive">{val}</span>'
        if s.startswith("-") or s.startswith("$-"):
            return f'<span class="negative">{val}</span>'
        return val

    return "\n".join(f"<tr><td>{k}</td><td>{_color(v)}</td></tr>" for k, v in rows)


def _build_reports_html(
    stats_pnls: dict,
    stats_returns: dict,
    stats_general: dict,
    positions_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    account_df: pd.DataFrame,
    chart: str,
) -> str:

    _sort_script = r"""
<script>
document.addEventListener('click', function(e) {
  var th = e.target.closest('th');
  if (!th || !th.closest('.sortable')) return;
  var table = th.closest('table');
  var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
  var tbody = table.querySelector('tbody') || table;
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
  if (!rows.length || rows[0].querySelectorAll('td').length <= idx) return;
  var asc = th._sortAsc !== true;
  th._sortAsc = asc;
  Array.prototype.forEach.call(table.querySelectorAll('th'), function(h) { if (h !== th) delete h._sortAsc; });
  rows.sort(function(a, b) {
    var av = (a.children[idx] || {}).textContent.trim();
    var bv = (b.children[idx] || {}).textContent.trim();
    var na = parseFloat(av.replace(/[$,%]/g, ''));
    var nb = parseFloat(bv.replace(/[$,%]/g, ''));
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
});
document.querySelectorAll('.table-wrap td').forEach(function(td) {
  var t = td.textContent.trim();
  if (/^\s*-/.test(t)) td.style.color = '#d32f2f';
  else if (/^\s*\+/.test(t)) td.style.color = '#2e7d32';
});
</script>
"""

    return f"""
<style>
.report-section {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  background: #f5f5f5;
  color: #222;
}}
.report-section h1 {{
  font-weight: 600;
  font-size: 24px;
  border-bottom: 2px solid #111;
  padding-bottom: 8px;
  margin-bottom: 20px;
}}
.report-section h2 {{
  font-weight: 600;
  font-size: 16px;
  margin-top: 28px;
  margin-bottom: 8px;
  color: #333;
}}
.table-wrap {{
  overflow-x: auto;
  overflow-y: auto;
  max-height: 420px;
  margin-bottom: 24px;
}}
.table-wrap table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  white-space: nowrap;
  background: #fff;
}}
.table-wrap th {{
  padding: 10px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #666;
  border-bottom: 2px solid #111;
  position: sticky;
  top: 0;
  z-index: 1;
  background: #f5f5f5;
}}
.table-wrap td {{
  padding: 8px 12px;
  border-bottom: 1px solid #e0e0e0;
}}
.table-wrap tr:nth-child(even) td {{
  background: #f9f9f9;
}}
.table-wrap tr:hover td {{
  background: #f0f0f0;
}}
.stats-table {{
  width: 100%;
  max-width: 520px;
  border-collapse: collapse;
  margin-bottom: 28px;
  background: #fff;
  font-size: 12px;
}}
.stats-table td {{
  padding: 7px 16px;
  border-bottom: 1px solid #e8e8e8;
}}
.stats-table tr:nth-child(even) td {{
  background: #fafafa;
}}
.stats-table td:first-child {{
  font-weight: 500;
  color: #555;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.04em;
}}
.stats-table td:last-child {{
  text-align: right;
  font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
  font-weight: 500;
}}
.stats-table tr:last-child td {{
  border-bottom: none;
}}
.toggle-btn {{
  background: #111;
  color: #fff;
  border: none;
  padding: 8px 24px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 12px;
}}
.toggle-btn:hover {{
  background: #333;
}}
#chart-section {{
  display: none;
}}
.positive {{
  color: #2e7d32;
}}
.negative {{
  color: #d32f2f;
}}
.sortable th {{
  cursor: pointer;
}}
.sortable th:hover {{
  color: #000;
}}
</style>
<div class="report-section">
  <h1>Backtest Report</h1>

  <h2>Portfolio Performance</h2>
  <table class="stats-table">
    {_process_stats(stats_pnls, stats_returns, stats_general)}
  </table>

  <h2>Positions ({len(positions_df)})</h2>
  <div class="table-wrap">{_df_to_html_table(positions_df)}</div>

  <h2>Fills ({len(fills_df)})</h2>
  <div class="table-wrap">{_df_to_html_table(fills_df)}</div>

  <h2>Orders ({len(orders_df)})</h2>
  <div class="table-wrap">{_df_to_html_table(orders_df)}</div>

  <h2>Account ({len(account_df)})</h2>
  <div class="table-wrap">{_df_to_html_table(account_df)}</div>
</div>

<div class="report-section">
  <h1>Chart</h1>
  <button class="toggle-btn" onclick="var s=document.getElementById('chart-section');s.style.display=s.style.display==='none'?'block':'none'">Toggle Chart</button>
  <div id="chart-section">
    {chart}
  </div>
</div>
{_sort_script}
"""


def print_report(
    engine: BacktestEngine,
    venue,
    title: str,
    pair: str = "",
    exchange: str = "",
    interval: str = "5m",
    open_browser: bool = True,
) -> None:
    print("\n========== BACKTEST COMPLETE ==========")

    stats_pnls = engine.portfolio.analyzer.get_performance_stats_pnls()
    stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
    stats_general = engine.portfolio.analyzer.get_performance_stats_general()

    print("\n--- Portfolio Performance ---")
    for k, v in {**stats_pnls, **stats_returns, **stats_general}.items():
        print(f"  {k}: {v}")

    positions_report = engine.trader.generate_positions_report()
    print(f"\n--- Positions Report ({len(positions_report)} rows) ---")
    print(positions_report.to_string(max_rows=20))

    fills_report = engine.trader.generate_fills_report()
    print(f"\n--- Fills Report ({len(fills_report)} rows) ---")
    print(fills_report.to_string(max_rows=20))

    orders_report = engine.trader.generate_orders_report()
    print(f"\n--- Orders Report ({len(orders_report)} rows) ---")
    print(orders_report.to_string(max_rows=20))

    account_report = engine.trader.generate_account_report(venue)
    print(f"\n--- Account Report ({len(account_report)} rows) ---")
    print(account_report.to_string(max_rows=10))

    run_id = engine.run_id
    Path("reports").mkdir(exist_ok=True)
    tearsheet_path = f"reports/tearsheet_{run_id}.html"

    print(f"\n--- Generating tearsheet ({run_id}) ---")
    create_tearsheet(
        engine,
        output_path=tearsheet_path,
        title=title,
    )

    html = Path(tearsheet_path).read_text()

    chart = _chart_block(
        positions_json=_positions_json(positions_report),
        pair=pair,
        exchange=exchange,
        interval=interval,
    )

    reports_html = _build_reports_html(
        stats_pnls=stats_pnls,
        stats_returns=stats_returns,
        stats_general=stats_general,
        positions_df=positions_report,
        fills_df=fills_report,
        orders_df=orders_report,
        account_df=account_report,
        chart=chart,
    )

    html = html.replace("</body>", reports_html + "\n</body>")
    Path(tearsheet_path).write_text(html)
    print("Report sections and chart injected into tearsheet.")

    print(f"Tearsheet saved to {tearsheet_path}")

    html_path = Path(tearsheet_path).resolve()
    if open_browser:
        webbrowser.open(f"file://{html_path}")

    print("\n========== DONE ==========")
