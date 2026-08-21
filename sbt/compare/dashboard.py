"""Multi-strategy performance comparison dashboard."""

import json
from pathlib import Path
import webbrowser
from typing import Any


def generate_comparison_dashboard(
    results: list[dict[str, Any]],
    output_path: str = "reports/compare.html",
    open_browser: bool = True,
) -> str:
    """Generate an interactive HTML dashboard comparing multiple backtest runs."""
    if not results:
        raise ValueError("Cannot generate comparison dashboard with empty results.")

    labels = []
    sharpe_vals = []
    pnl_vals = []
    trades_vals = []

    for r in results:
        job_id = r.get("job_id", "N/A")
        stats = r.get("stats", {})
        strat_name = stats.get("Strategy", job_id)
        label = f"{strat_name} ({job_id})"
        labels.append(label)

        sharpe_vals.append(r.get("sharpe_ratio") or 0.0)
        pnl_vals.append(r.get("pnl") or 0.0)
        trades_vals.append(r.get("num_trades") or 0)

    # Collect all unique metric keys
    all_keys = set()
    for r in results:
        all_keys.update(r.get("stats", {}).keys())

    # Sort keys logically
    priority_keys = [
        "PnL (total)",
        "PnL% (total)",
        "Sharpe Ratio (252 days)",
        "Sortino Ratio (252 days)",
        "Calmar Ratio",
        "Annualized Return",
        "Max Drawdown",
        "Win Rate",
        "Profit Factor",
        "Total Trades",
    ]
    other_keys = sorted([k for k in all_keys if k not in priority_keys and "config" not in k.lower()])
    ordered_keys = [k for k in priority_keys if k in all_keys] + other_keys

    # Build comparison table rows
    table_rows_html = ""
    for k in ordered_keys:
        cells = []
        for r in results:
            val = r.get("stats", {}).get(k, "—")
            if isinstance(val, float):
                val_str = f"{val:.2f}"
            else:
                val_str = str(val)

            # Color coding
            cell_class = ""
            if "pnl" in k.lower() or "return" in k.lower():
                if val_str.startswith("+") or val_str.startswith("$+"):
                    cell_class = 'class="positive"'
                elif val_str.startswith("-") or val_str.startswith("$-"):
                    cell_class = 'class="negative"'

            cells.append(f"<td {cell_class}>{val_str}</td>")

        table_rows_html += f"<tr><td><strong>{k}</strong></td>{''.join(cells)}</tr>"

    headers_html = "".join(f"<th>{l}</th>" for l in labels)

    chart_data_json = json.dumps({
        "labels": labels,
        "sharpe": sharpe_vals,
        "pnl": pnl_vals,
        "trades": trades_vals,
    })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SBT Strategy Comparison Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 24px;
      background: #0f172a;
      color: #f8fafc;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 26px;
      margin-bottom: 4px;
      color: #38bdf8;
    }}
    .subtitle {{
      color: #94a3b8;
      margin-bottom: 24px;
      font-size: 14px;
    }}
    .card {{
      background: #1e293b;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .card h2 {{
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 16px;
      color: #f1f5f9;
      border-bottom: 1px solid #334155;
      padding-bottom: 8px;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .chart-box {{
      background: #1e293b;
      border-radius: 8px;
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      background: #0f172a;
      padding: 10px 12px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid #334155;
    }}
    tr:hover td {{
      background: #273549;
    }}
    .positive {{ color: #4ade80; font-weight: 500; }}
    .negative {{ color: #f87171; font-weight: 500; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Multi-Strategy Comparison Dashboard</h1>
    <div class="subtitle">Comparing {len(results)} backtest runs</div>

    <div class="charts-grid">
      <div class="chart-box">
        <div id="chart-sharpe" style="height: 320px;"></div>
      </div>
      <div class="chart-box">
        <div id="chart-pnl" style="height: 320px;"></div>
      </div>
    </div>

    <div class="card">
      <h2>Side-by-Side Performance Metrics</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              {headers_html}
            </tr>
          </thead>
          <tbody>
            {table_rows_html}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    var data = {chart_data_json};

    // Sharpe Chart
    Plotly.newPlot('chart-sharpe', [{{
      x: data.labels,
      y: data.sharpe,
      type: 'bar',
      marker: {{ color: '#38bdf8' }}
    }}], {{
      title: 'Sharpe Ratio (252 days)',
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#1e293b',
      font: {{ color: '#f8fafc' }},
      margin: {{ l: 40, r: 20, t: 40, b: 60 }}
    }}, {{responsive: true}});

    // PnL Chart
    Plotly.newPlot('chart-pnl', [{{
      x: data.labels,
      y: data.pnl,
      type: 'bar',
      marker: {{
        color: data.pnl.map(function(v) {{ return v >= 0 ? '#4ade80' : '#f87171'; }})
      }}
    }}], {{
      title: 'Net Total PnL ($)',
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#1e293b',
      font: {{ color: '#f8fafc' }},
      margin: {{ l: 50, r: 20, t: 40, b: 60 }}
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")

    if open_browser:
        try:
            webbrowser.open(f"file://{out_file.resolve()}")
        except Exception:
            pass

    return str(out_file.resolve())
