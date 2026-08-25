"""Generates interactive HTML reports for Optuna optimization studies."""

import json
import webbrowser
from pathlib import Path

import optuna


def _write_and_open(html_content: str, output_path: str, open_browser: bool) -> str:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")

    if open_browser:
        try:
            webbrowser.open(f"file://{out_file.resolve()}")
        except Exception:
            pass

    return str(out_file.resolve())


_CSS = """
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 24px;
      background: #0f172a;
      color: #f8fafc;
    }
    .container {
      max-width: 1300px;
      margin: 0 auto;
    }
    h1 {
      font-size: 26px;
      margin-bottom: 4px;
      color: #38bdf8;
    }
    .subtitle {
      color: #94a3b8;
      margin-bottom: 24px;
      font-size: 14px;
    }
    .card {
      background: #1e293b;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    .card h2 {
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 16px;
      color: #f1f5f9;
      border-bottom: 1px solid #334155;
      padding-bottom: 8px;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .stat-box {
      background: #0f172a;
      padding: 16px;
      border-radius: 6px;
      border-left: 4px solid #38bdf8;
    }
    .stat-val {
      font-size: 24px;
      font-weight: bold;
      color: #38bdf8;
    }
    .stat-lbl {
      font-size: 12px;
      color: #94a3b8;
      text-transform: uppercase;
      margin-top: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    th {
      background: #0f172a;
      padding: 10px 12px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #334155;
    }
    tr:hover td {
      background: #273549;
    }
    .positive { color: #4ade80; font-weight: 500; }
    .negative { color: #f87171; font-weight: 500; }
    #plot3d, #plot2d {
      width: 100%;
      height: 520px;
    }
"""


def generate_pareto_report(
    study: optuna.Study,
    strategy_name: str,
    output_path: str = "reports/pareto_report.html",
    open_browser: bool = True,
    primary_label: str = "Sharpe Ratio",
    primary_short: str = "Sharpe",
) -> str:
    """Generate an interactive HTML report visualizing the Pareto front."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trials = study.best_trials

    best_trial_numbers = {t.number for t in best_trials}

    trials_data = []
    for t in trials:
        primary, trades, pnl = t.values if t.values else (0.0, 0, 0.0)
        is_pareto = t.number in best_trial_numbers
        trials_data.append(
            {
                "number": t.number,
                "primary": round(primary, 3) if primary else 0.0,
                "trades": int(trades) if trades else 0,
                "pnl": round(pnl, 2) if pnl else 0.0,
                "params": t.params,
                "is_pareto": is_pareto,
            }
        )

    trials_json = json.dumps(trials_data)

    # Collect parameter names
    param_names = (
        sorted(list(study.best_trials[0].params.keys())) if study.best_trials else []
    )

    pareto_rows_html = ""
    for t in sorted(
        best_trials, key=lambda x: x.values[0] if x.values else 0, reverse=True
    ):
        primary, trades, pnl = t.values if t.values else (0.0, 0, 0.0)
        param_cells = "".join(f"<td>{t.params.get(p, '-')}</td>" for p in param_names)
        pareto_rows_html += f"""
        <tr>
            <td><strong>#{t.number}</strong></td>
            <td class="positive">{primary:.2f}</td>
            <td>{int(trades)}</td>
            <td class="{"positive" if pnl >= 0 else "negative"}">${pnl:+,.2f}</td>
            {param_cells}
        </tr>
        """

    param_headers_html = "".join(f"<th>{p}</th>" for p in param_names)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Optuna Multi-Objective Optimization — {strategy_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>
  <div class="container">
    <h1>Optuna Multi-Objective Optimization — {strategy_name}</h1>
    <div class="subtitle">Study: {study.study_name} | Total Trials: {len(trials)} | Pareto Frontier Size: {len(best_trials)}</div>

    <div class="stats-grid">
      <div class="stat-box">
        <div class="stat-val">{len(trials)}</div>
        <div class="stat-lbl">Completed Trials</div>
      </div>
      <div class="stat-box" style="border-left-color: #4ade80;">
        <div class="stat-val" style="color: #4ade80;">{len(best_trials)}</div>
        <div class="stat-lbl">Pareto-Optimal Solutions</div>
      </div>
      <div class="stat-box" style="border-left-color: #fbbf24;">
        <div class="stat-val" style="color: #fbbf24;">{max([t.values[0] for t in trials if t.values] or [0]):.2f}</div>
        <div class="stat-lbl">Max {primary_label}</div>
      </div>
      <div class="stat-box" style="border-left-color: #c084fc;">
        <div class="stat-val" style="color: #c084fc;">${max([t.values[2] for t in trials if t.values] or [0]):+,.2f}</div>
        <div class="stat-lbl">Max Net PnL</div>
      </div>
    </div>

    <div class="card">
      <h2>Interactive 3D Pareto Frontier ({primary_label} × Trades × PnL)</h2>
      <div id="plot3d"></div>
    </div>

    <div class="card">
      <h2>Pareto-Optimal Set (Non-Dominated Trade-Offs)</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Trial</th>
              <th>{primary_short}</th>
              <th>Trades</th>
              <th>Net PnL</th>
              {param_headers_html}
            </tr>
          </thead>
          <tbody>
            {pareto_rows_html}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    var data = {trials_json};

    var paretoX = [], paretoY = [], paretoZ = [], paretoText = [];
    var otherX = [], otherY = [], otherZ = [], otherText = [];

    data.forEach(function(d) {{
      var hover = "Trial #" + d.number + "<br>{primary_short}: " + d.primary + "<br>Trades: " + d.trades + "<br>PnL: $" + d.pnl + "<br>" + JSON.stringify(d.params);
      if (d.is_pareto) {{
        paretoX.push(d.primary);
        paretoY.push(d.trades);
        paretoZ.push(d.pnl);
        paretoText.push(hover);
      }} else {{
        otherX.push(d.primary);
        otherY.push(d.trades);
        otherZ.push(d.pnl);
        otherText.push(hover);
      }}
    }});

    var traceOther = {{
      x: otherX, y: otherY, z: otherZ,
      mode: 'markers',
      type: 'scatter3d',
      text: otherText,
      hoverinfo: 'text',
      marker: {{
        size: 4,
        color: '#64748b',
        opacity: 0.6
      }},
      name: 'All Trials'
    }};

    var tracePareto = {{
      x: paretoX, y: paretoY, z: paretoZ,
      mode: 'markers',
      type: 'scatter3d',
      text: paretoText,
      hoverinfo: 'text',
      marker: {{
        size: 8,
        color: '#4ade80',
        symbol: 'diamond',
        line: {{ color: '#ffffff', width: 1 }}
      }},
      name: 'Pareto Optimal'
    }};

    var layout = {{
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#1e293b',
      font: {{ color: '#f8fafc', family: 'Inter, sans-serif' }},
      scene: {{
        xaxis: {{ title: '{primary_label}', gridcolor: '#334155', zerolinecolor: '#475569' }},
        yaxis: {{ title: 'Num Trades', gridcolor: '#334155', zerolinecolor: '#475569' }},
        zaxis: {{ title: 'Net PnL ($)', gridcolor: '#334155', zerolinecolor: '#475569' }}
      }},
      margin: {{ l: 0, r: 0, b: 0, t: 0 }},
      legend: {{ x: 0.05, y: 0.95 }}
    }};

    Plotly.newPlot('plot3d', [traceOther, tracePareto], layout, {{responsive: true}});
  </script>
</body>
</html>
"""

    return _write_and_open(html_content, output_path, open_browser)


def generate_sqn_report(
    study: optuna.Study,
    strategy_name: str,
    output_path: str = "reports/sqn_report.html",
    open_browser: bool = True,
) -> str:
    """Generate an HTML report for a single-objective SQN study."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials if t.value is not None]
    best = max(trials, key=lambda t: t.value or float("-inf"), default=None)

    param_names = sorted(best.params.keys()) if best else []

    top_rows_html = ""
    for t in sorted(trials, key=lambda x: x.value or float("-inf"), reverse=True)[:10]:
        params_cells = "".join(f"<td>{t.params.get(p, '-')}</td>" for p in param_names)
        top_rows_html += f"""
        <tr>
            <td><strong>#{t.number}</strong></td>
            <td class="positive">{t.value:+.4f}</td>
            {params_cells}
        </tr>
        """
    param_headers_html = "".join(f"<th>{p}</th>" for p in param_names)

    history_data = json.dumps(
        [
            {"number": t.number, "sqn": round(t.value, 4) if t.value else 0.0}
            for t in trials
        ]
    )
    best_number = best.number if best else -1

    mean_sqn = sum(values) / len(values) if values else 0.0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Optuna SQN Optimization — {strategy_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>
  <div class="container">
    <h1>Optuna SQN Optimization — {strategy_name}</h1>
    <div class="subtitle">Study: {study.study_name} | Objective: Van Tharp System Quality Number (per-trade returns)</div>

    <div class="stats-grid">
      <div class="stat-box">
        <div class="stat-val">{len(trials)}</div>
        <div class="stat-lbl">Completed Trials</div>
      </div>
      <div class="stat-box" style="border-left-color: #4ade80;">
        <div class="stat-val" style="color: #4ade80;">{(best.value if best else 0):+.4f}</div>
        <div class="stat-lbl">Best SQN (Trial #{best_number})</div>
      </div>
      <div class="stat-box" style="border-left-color: #fbbf24;">
        <div class="stat-val" style="color: #fbbf24;">{mean_sqn:+.4f}</div>
        <div class="stat-lbl">Mean SQN</div>
      </div>
    </div>

    <div class="card">
      <h2>Best Parameters</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr><th>Parameter</th><th>Value</th></tr>
          </thead>
          <tbody>
            {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in (best.params.items() if best else []))}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Optimization History (SQN per Trial)</h2>
      <div id="plot2d"></div>
    </div>

    <div class="card">
      <h2>Top 10 Trials by SQN</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Trial</th>
              <th>SQN</th>
              {param_headers_html}
            </tr>
          </thead>
          <tbody>
            {top_rows_html}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    var history = {history_data};
    var bestNumber = {best_number};

    var xs = history.map(function(d) {{ return d.number; }});
    var ys = history.map(function(d) {{ return d.sqn; }});
    var colors = history.map(function(d) {{
      return d.number === bestNumber ? '#4ade80' : '#38bdf8';
    }});
    var sizes = history.map(function(d) {{
      return d.number === bestNumber ? 12 : 7;
    }});

    Plotly.newPlot('plot2d', [{{
      x: xs, y: ys,
      mode: 'lines+markers',
      line: {{ color: '#334155', width: 1 }},
      marker: {{ color: colors, size: sizes }},
      hovertemplate: 'Trial #%{{x}}<br>SQN: %{{y:+.4f}}<extra></extra>'
    }}], {{
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#1e293b',
      font: {{ color: '#f8fafc', family: 'Inter, sans-serif' }},
      xaxis: {{ title: 'Trial', gridcolor: '#334155', zerolinecolor: '#475569' }},
      yaxis: {{ title: 'System Quality Number', gridcolor: '#334155', zerolinecolor: '#475569' }},
      margin: {{ l: 60, r: 20, b: 50, t: 10 }},
      showlegend: false
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""

    return _write_and_open(html_content, output_path, open_browser)
