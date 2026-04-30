function Topbar() {
  return (
    <header className="topbar">
      <a className="brand" href="#" aria-label="MLB Intelligence home">
        <img src="../../assets/logo.svg" alt="" />
        <span>MLB Intelligence</span>
      </a>
      <nav className="nav-links" aria-label="Primary">
        <a href="#public">Public analytics</a>
        <a href="#contact">Contact</a>
        <a href="#privacy">Privacy</a>
        <a href="#login">Log in</a>
      </nav>
    </header>
  );
}
window.Topbar = Topbar;
