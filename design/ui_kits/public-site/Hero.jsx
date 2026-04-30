function Hero() {
  return (
    <>
      <section className="hero-orbit" aria-hidden="true">
        <div className="orbit orbit-a"></div>
        <div className="orbit orbit-b"></div>
        <div className="orbit orbit-c"></div>
      </section>
      <section className="ps-hero">
        <div className="hero-copy hero-copy-stage">
          <span className="eyebrow">Model insights</span>
          <h1>Public MLB model results, updated daily.</h1>
          <p className="ps-hero-copy">
            MLB Intelligence is a public audit trail for a model-driven baseball
            betting workflow. Start with live ROI, calibration, market comparison,
            and recent settled wagers.
          </p>
          <div className="hero-actions">
            <a className="btn-primary" href="#public">View Public Analytics</a>
            <a className="btn-secondary" href="#register">Request Access</a>
          </div>
          <div className="hero-badges">
            <span>Public proof</span>
            <span>Private operator controls</span>
            <span>Live model metrics</span>
          </div>
        </div>
        <aside className="hero-demo" aria-label="Overview">
          <div className="hero-demo-grid">
            <article className="demo-panel demo-panel-primary">
              <div className="demo-kicker">Live public snapshot</div>
              <div className="demo-scoreline">
                <div><span>ROI</span><strong>+12.4%</strong></div>
                <div><span>Bets tracked</span><strong>1,247</strong></div>
              </div>
              <p className="demo-note">Loaded from public endpoints.</p>
              <div className="demo-bars" aria-hidden="true">
                <span style={{'--bar': '74%'}}></span>
                <span style={{'--bar': '58%'}}></span>
                <span style={{'--bar': '83%'}}></span>
                <span style={{'--bar': '66%'}}></span>
                <span style={{'--bar': '91%'}}></span>
                <span style={{'--bar': '61%'}}></span>
              </div>
            </article>
            <article className="demo-panel demo-panel-float">
              <span className="demo-kicker">Public analytics</span>
              <strong>Calibration, ROI, market comparison</strong>
              <p>The public surface is built to show the receipts.</p>
            </article>
            <article className="demo-panel demo-panel-float alt">
              <span className="demo-kicker">Private workspace</span>
              <strong>Live betting state, exposure, bankroll</strong>
              <p>Operational controls stay behind approval.</p>
            </article>
          </div>
        </aside>
      </section>
    </>
  );
}
window.Hero = Hero;
