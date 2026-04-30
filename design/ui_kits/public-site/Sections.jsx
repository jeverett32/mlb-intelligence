function MetricsGrid({ metrics }) {
  const items = metrics || [
    { label: 'Total bets', value: '1,247' },
    { label: 'Win rate', value: '52.4%' },
    { label: 'ROI', value: '+12.4%' },
    { label: 'Model accuracy', value: '54.1%' },
  ];
  return (
    <section className="trust-metrics">
      <div className="metrics-grid">
        {items.map((m, i) => (
          <div key={i} className="metric-card">
            <span className="metric-label">{m.label}</span>
            <span className="metric-value">{m.value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function ProcessSteps() {
  const steps = [
    { n: 1, h: 'Data collection', p: 'Game history, team form, player inputs, weather, and pricing feeds collected into one repeatable pipeline.' },
    { n: 2, h: 'Model prediction', p: 'Probability estimates generated from engineered features and checked against historical calibration.' },
    { n: 3, h: 'Market analysis', p: 'Model view compared with market-implied pricing to surface meaningful divergence.' },
    { n: 4, h: 'Risk management', p: 'Position sizing and account-level controls determine whether a signal turns into a live order.' },
  ];
  return (
    <section className="how-it-works">
      <div className="section-header section-header-left">
        <span className="eyebrow">Process</span>
        <h2>How it works</h2>
      </div>
      <div className="process-grid">
        {steps.map(s => (
          <div key={s.n} className="process-step">
            <div className="step-number">{s.n}</div>
            <h3>{s.h}</h3>
            <p>{s.p}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function AnalyticsKPIs() {
  return (
    <section className="analytics-overview-grid">
      <article className="analytics-overview-card">
        <span className="analytics-card-label">Model Accuracy</span>
        <strong>54.1%</strong>
        <p>All predicted games, measured against actual outcomes.</p>
      </article>
      <article className="analytics-overview-card">
        <span className="analytics-card-label">Market Accuracy</span>
        <strong>51.6%</strong>
        <p>Sportsbook baseline, tracked against the same settled sample.</p>
      </article>
      <article className="analytics-overview-card">
        <span className="analytics-card-label">Bet ROI</span>
        <strong style={{color: 'var(--success)'}}>+12.4%</strong>
        <p>Return on real wagers from the model recommendation set.</p>
      </article>
    </section>
  );
}

function ReceiptsTable() {
  const rows = [
    { d: 'Apr 28', m: 'NYY @ BOS', s: 'NYY', r: 'W', mp: '58.2%', kp: '52.0%', e: '6.2 pts', ret: '+8.1%' },
    { d: 'Apr 27', m: 'LAD @ SDP', s: 'LAD', r: 'L', mp: '61.5%', kp: '57.3%', e: '4.2 pts', ret: '−4.0%' },
    { d: 'Apr 27', m: 'HOU @ SEA', s: 'HOU', r: 'W', mp: '54.8%', kp: '50.0%', e: '4.8 pts', ret: '+5.4%' },
    { d: 'Apr 26', m: 'CHC @ PHI', s: 'PHI', r: 'W', mp: '56.1%', kp: '52.4%', e: '3.7 pts', ret: '+6.2%' },
    { d: 'Apr 26', m: 'ATL @ MIA', s: 'ATL', r: 'L', mp: '63.2%', kp: '60.1%', e: '3.1 pts', ret: '−3.0%' },
  ];
  return (
    <section className="analytics-section analytics-section-receipts">
      <div className="analytics-section-header">
        <div>
          <span className="eyebrow">Recent Settled Bets</span>
          <h2>The receipts</h2>
        </div>
      </div>
      <div className="analytics-table-wrap">
        <table className="analytics-table">
          <thead><tr>
            <th>Date</th><th>Matchup</th><th>Side</th><th>Result</th>
            <th>Model</th><th>Market</th><th>Edge</th><th>Return</th>
          </tr></thead>
          <tbody>
            {rows.map((b, i) => (
              <tr key={i}>
                <td>{b.d}</td><td>{b.m}</td><td>{b.s}</td>
                <td><span className={`result-pill ${b.r === 'W' ? 'win' : 'loss'}`}>{b.r}</span></td>
                <td>{b.mp}</td><td>{b.kp}</td><td>{b.e}</td>
                <td className={b.ret.startsWith('+') ? 'positive' : 'negative'}>{b.ret}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="public-footer">
      <span>Metrics update from public endpoints.</span>
      <nav aria-label="Footer">
        <a href="#public">Analytics</a>
        <a href="#contact">Contact</a>
        <a href="#privacy">Privacy</a>
        <a href="#login">Log in</a>
      </nav>
    </footer>
  );
}

Object.assign(window, { MetricsGrid, ProcessSteps, AnalyticsKPIs, ReceiptsTable, Footer });
