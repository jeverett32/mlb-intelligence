// Shared theme persistence for public-facing surfaces (landing, /public, login, register).
// Must load without defer so data-theme is set before the body paints.

(function () {
  document.documentElement.setAttribute(
    'data-theme',
    localStorage.getItem('theme') || 'auto'
  );
})();

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

  // Load live metrics for landing page
  loadLiveMetrics();
});

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
