function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem('theme', mode);
  var sel = document.getElementById('theme-select');
  if (sel) sel.value = mode;
  document.dispatchEvent(new CustomEvent('themechange', { detail: { mode: mode } }));
}

document.addEventListener('DOMContentLoaded', function () {
  var sel = document.getElementById('theme-select');
  if (sel) sel.value = localStorage.getItem('theme') || 'auto';

  initRevealAnimations();
  initTiltPanels();

  // Load live metrics for landing page
  loadLiveMetrics();
});

function initRevealAnimations() {
  var sections = document.querySelectorAll('[data-reveal]');
  if (!sections.length) return;

  if (!('IntersectionObserver' in window)) {
    sections.forEach(function(section) {
      section.classList.add('is-visible');
    });
    return;
  }

  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.12,
    rootMargin: '0px 0px -8% 0px'
  });

  sections.forEach(function(section, index) {
    section.style.transitionDelay = Math.min(index * 60, 240) + 'ms';
    observer.observe(section);
  });
}

function initTiltPanels() {
  var panels = document.querySelectorAll('[data-tilt]');
  if (
    !panels.length ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches ||
    window.matchMedia('(pointer: coarse)').matches ||
    window.matchMedia('(hover: none)').matches
  ) {
    return;
  }

  panels.forEach(function(panel) {
    var rect = null;
    var rafId = 0;
    var nextX = 0;
    var nextY = 0;

    function measure() {
      rect = panel.getBoundingClientRect();
    }

    function render() {
      rafId = 0;
      if (!rect) measure();
      var px = (nextX - rect.left) / rect.width;
      var py = (nextY - rect.top) / rect.height;
      var rx = (0.5 - py) * 8;
      var ry = (px - 0.5) * 10;
      panel.style.transform = 'rotateX(' + rx.toFixed(2) + 'deg) rotateY(' + ry.toFixed(2) + 'deg)';
    }

    measure();
    panel.addEventListener('mouseenter', measure);
    window.addEventListener('resize', measure);
    panel.addEventListener('mousemove', function(event) {
      nextX = event.clientX;
      nextY = event.clientY;
      if (!rafId) {
        rafId = window.requestAnimationFrame(render);
      }
    });

    panel.addEventListener('mouseleave', function() {
      if (rafId) {
        window.cancelAnimationFrame(rafId);
        rafId = 0;
      }
      panel.style.transform = '';
    });
  });
}

function loadLiveMetrics() {
  var metricsSection = document.querySelector('[data-live-metrics]');
  var performanceSection = document.querySelector('#performance-summary');

  if (!metricsSection && !performanceSection) return;

  fetch('/api/public/summary')
    .then(function(response) {
      if (!response.ok) throw new Error('Failed to load metrics');
      return response.json();
    })
    .then(function(data) {
      // Update trust metrics strip
      if (metricsSection) {
        updateMetric('total_bets', data.performance.total_bets);
        updateMetric('accuracy', (data.performance.accuracy * 100).toFixed(1) + '%');
        updateMetric('roi_pct', (data.performance.roi_pct >= 0 ? '+' : '') + data.performance.roi_pct.toFixed(1) + '%');
        updateMetric('model_accuracy', (data.model_accuracy.accuracy * 100).toFixed(1) + '%');
      }

      // Update performance summary section
      if (performanceSection) {
        updatePerformanceMetric('perf', 'accuracy', (data.performance.accuracy * 100).toFixed(1) + '%');
        updatePerformanceMetric('perf', 'roi_pct', (data.performance.roi_pct >= 0 ? '+' : '') + data.performance.roi_pct.toFixed(1) + '%');
        updatePerformanceMetric('perf', 'total_bets', data.performance.total_bets);
        updatePerformanceMetric('model', 'accuracy', (data.model_accuracy.accuracy * 100).toFixed(1) + '%');
        updatePerformanceMetric('model', 'market_accuracy', (data.model_accuracy.market_accuracy * 100).toFixed(1) + '%');
        updatePerformanceMetric('model', 'total', data.model_accuracy.total);
      }
    })
    .catch(function(error) {
      console.warn('Could not load live metrics:', error);
      // Keep default "—" values on error
    });
}

function updateMetric(key, value) {
  var element = document.querySelector('[data-metric="' + key + '"]');
  if (element) element.textContent = value;
}

function updatePerformanceMetric(prefix, key, value) {
  var element = document.querySelector('[data-' + prefix + '="' + key + '"]');
  if (element) element.textContent = value;
}
