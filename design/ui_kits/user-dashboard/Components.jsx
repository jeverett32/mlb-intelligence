function TopNav({ active, onChange, view, onView }) {
  const tabs = ['Overview', 'Games', 'Teams', 'Bets', 'Performance'];
  return (
    <header className="d-topbar">
      <div className="d-topbar-row">
        <div className="d-brand">
          <span className="d-logo">MLB Intelligence</span>
          <span className="d-season">2026 Season</span>
        </div>
        <div className="d-balance-wrap">
          <span className="d-balance"><span className="lab">Balance:</span> <strong>$11.05</strong></span>
          <button className="d-gear" aria-label="Settings">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>
          </button>
        </div>
      </div>
      <div className="d-tabs-row">
        <nav className="d-tabs" aria-label="Sections">
          {tabs.map(t => (
            <a key={t} href={`#${t.toLowerCase()}`} className={`d-tab ${t === active ? 'active' : ''}`} onClick={e => { e.preventDefault(); onChange(t); }}>{t}</a>
          ))}
        </nav>
        <div className="d-view-toggle">
          <span className="lab">Bet view</span>
          <div className="seg" role="tablist">
            <button className={view === 'Paper' ? 'on' : ''} onClick={() => onView('Paper')}>Paper</button>
            <button className={view === 'Live' ? 'on' : ''} onClick={() => onView('Live')}>Live</button>
          </div>
        </div>
      </div>
    </header>
  );
}

function SnapshotCard() {
  return (
    <article className="d-snapshot">
      <span className="d-kicker">Operational snapshot</span>
      <h2 className="d-snap-h">No open positions right now.</h2>
      <p className="d-snap-p">Next up: <strong>LAA at CHW</strong> at Apr 29, 1:10 AM.</p>
      <div className="d-snap-row">
        <div className="d-mini">
          <span className="lab">Next game</span>
          <strong>LAA @ CHW</strong>
        </div>
        <div className="d-mini">
          <span className="lab">Open exposure</span>
          <strong>$0.00</strong>
        </div>
      </div>
    </article>
  );
}

function ChartCard() {
  // hand-built SVG chart with the same shape as the reference
  return (
    <article className="d-chart">
      <header><span className="d-kicker">Bankroll history</span></header>
      <div className="d-chart-legend">
        <span><i style={{background:'transparent', border:'1px dashed var(--accent)'}}></i> Available cash</span>
        <span><i style={{background:'var(--accent)'}}></i> Estimated bankroll</span>
      </div>
      <svg viewBox="0 0 520 200" className="d-chart-svg" preserveAspectRatio="none">
        {[0,1,2,3,4,5,6].map(i => (
          <line key={i} x1="44" x2="510" y1={20 + i*26} y2={20 + i*26} stroke="var(--border)" strokeWidth="0.5" />
        ))}
        {['$11.50','$11.00','$10.50','$10.00','$9.50','$9.00','$8.50','$7.50'].map((t, i) => (
          <text key={t} x="38" y={24 + i*26} fontSize="9" fill="var(--text-muted)" textAnchor="end" fontFamily="ui-monospace">{t}</text>
        ))}
        {['2026-04-21','2026-04-23','2026-04-25','2026-04-27','2026-04-29','now'].map((t,i) => (
          <text key={t} x={50 + i*92} y="195" fontSize="9" fill="var(--text-muted)" fontFamily="ui-monospace">{t}</text>
        ))}
        <path d="M 44 64 L 110 70 L 176 92 L 240 124 L 306 152 L 370 130 L 432 70 L 510 28" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        <path d="M 44 66 L 110 72 L 176 94 L 240 126 L 306 154 L 370 132 L 432 72 L 510 30" fill="none" stroke="var(--accent)" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.6" />
      </svg>
    </article>
  );
}

function KPIStrip() {
  const items = [
    { label: 'Current balance', value: '$11.05', sub: 'Estimated bankroll is available cash ($11.05) plus $0.00 in open positions.' },
    { label: 'Live betting', value: 'ON', tone: 'pos', sub: 'Your account can place live orders.' },
    { label: 'Open bets', value: '0', sub: 'Current unresolved wagers with dollars committed.' },
    { label: 'Season ROI', value: '+37.93%', tone: 'pos', sub: 'Realized return on settled bets.' },
    { label: 'Model vs market', value: '−4.3 pts', tone: 'neg', sub: 'Model 48.9% vs market 53.3%.' },
  ];
  return (
    <section className="d-kpi-strip" aria-label="Account metrics">
      {items.map(k => (
        <article key={k.label} className="d-kpi">
          <span className="d-kicker">{k.label}</span>
          <strong className={`d-kpi-val ${k.tone || ''}`}>{k.value}</strong>
          <p>{k.sub}</p>
        </article>
      ))}
    </section>
  );
}

function PositionsPanel() {
  return (
    <section className="d-panel">
      <header className="d-panel-head"><span className="d-kicker">Open positions</span></header>
      <div className="d-empty">No open positions</div>
    </section>
  );
}

Object.assign(window, { TopNav, SnapshotCard, ChartCard, KPIStrip, PositionsPanel });
