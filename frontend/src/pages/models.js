// Models: model cards, evaluation with baselines, data sources and their freshness.
import { api } from '../api.js';
import { h, clear, badge, trainedBadge, kindBadge, fmt, pct, ago, errorBox, loading, notice, put } from '../ui.js';

const num = (x, d = 1) => (x === undefined || x === null ? '–' : fmt(x, d));
const ci = (m) => (m && m.mae_ci95 ? `${num(m.mae_ci95[0], 2)} to ${num(m.mae_ci95[1], 2)}` : '–');

function metricTable(rows) {
  return h('div', { class: 'tscroll' }, h('table', { class: 'tbl' },
    h('thead', null, h('tr', null, ['', 'n', 'MAE (min)', '95% CI', 'RMSE', 'F1 (>15 min)'].map((t, i) => h('th', { class: i ? 'r' : '' }, t)))),
    h('tbody', null, rows.filter(([, m]) => m && m.mae !== undefined).map(([name, m]) => h('tr', null,
      h('td', null, name), h('td', { class: 'r mono' }, m.n), h('td', { class: 'r mono' }, num(m.mae, 2)), h('td', { class: 'r mono mute' }, ci(m)), h('td', { class: 'r mono' }, num(m.rmse, 2)),
      h('td', { class: 'r mono' }, num(m.f1_gt15 ?? m.f1_arr15, 2)))))));
}

const verdict = (v) => (v.better ? `better by ${num((1 - v.rmse_ratio) * 100, 1)}%${v.rmse_ratio > 0.995 ? ' (negligible)' : ''}` : v.ci95[0] > 1 ? `worse by ${num((v.rmse_ratio - 1) * 100, 1)}%` : 'not distinguishable');
// RMSE(model) / RMSE(baseline) with a paired block-bootstrap interval over days: below 1 is better, and only a whole interval below 1 counts.
function versus(title, vs) {
  if (!vs) return [];
  return [h('h3', null, title), h('dl', { class: 'kv' }, Object.entries(vs).flatMap(([k, v]) => [h('dt', null, k.replace(/_/g, ' ')), h('dd', { class: 'mono' }, `${num(v.rmse_ratio, 3)} (95% CI ${num(v.ci95[0], 3)} to ${num(v.ci95[1], 3)}, ${v.n_days} days): ${verdict(v)}`)]))];
}

const reduced = (c) => c.trained_on === 'real' && c.source_meta && Array.isArray(c.source_meta.months) && c.source_meta.months.length < 24;

function card(name, m) {
  const parts = [h('div', { class: 'row' }, h('h2', null, name), kindBadge(m.kind), m.trained_on === 'n/a' ? h('span', { class: 'badge b-kind' }, 'no training data') : trainedBadge(m.trained_on)), h('p', { class: 'small' }, m.name)];
  if (m.label) parts.push(h('p', null, h('b', null, m.label), m.kind === 'prior' ? '. No free real strike labels exist, so this is an ecology prior: season, local solar time, latitude and flyway, blended with eBird abundance when real data is available.' : '. Great-circle path, range-aware stops and jet-wind time: computed, not learned.'));
  if (m.metrics) {
    const mm = m.metrics;
    if (mm.y_dep) {
      parts.push(h('h3', null, 'Departure delay per airport-hour (test block)'), metricTable([['GCN', mm.y_dep], ['Historical mean', m.baselines.y_dep.hist_mean], ['LightGBM without graph', m.baselines.y_dep.lgbm_no_graph]]));
      parts.push(h('p', { class: 'small' }, m.graph_helps ? 'The graph helps: the GCN RMSE beats the same features without the graph, whole interval below 1.' : 'The GCN does NOT beat the no-graph LightGBM on RMSE (interval not below 1), so no claim is made that the graph helps.'));
      parts.push(h('h3', null, 'Late-aircraft delay'), metricTable([['GCN', mm.y_late], ['Historical mean', m.baselines.y_late.hist_mean], ['LightGBM without graph', m.baselines.y_late.lgbm_no_graph]]));
      parts.push(...versus('RMSE ratio vs baselines: departure delay', m.vs_baseline && m.vs_baseline.y_dep), ...versus('RMSE ratio vs baselines: late-aircraft delay', m.vs_baseline && m.vs_baseline.y_late));
    } else {
      parts.push(metricTable([[name, mm], ...Object.entries(m.baselines || {}).map(([k, v]) => [k.replace(/_/g, ' '), v])]));
      parts.push(...versus('RMSE ratio vs baselines (test block)', m.vs_baseline));
      if (m.oof) parts.push(...versus(`Same on train out-of-fold rows (n = ${m.oof.n}; used to pick variants, so not an unbiased test)`, m.oof.vs_baseline));
    }
  }
  if (m.kind === 'stacked') {
    const w = m.weights_pooled.weights;
    parts.push(h('h3', null, 'Stacker weights (pooled)'), h('dl', { class: 'kv' }, Object.entries(w).flatMap(([k, v]) => [h('dt', null, k), h('dd', null, num(v, 3))]), h('dt', null, 'baseline'), h('dd', null, `${num(m.weights_pooled.baseline, 2)} min`)));
    parts.push(h('h3', null, 'By horizon'), h('dl', { class: 'kv' }, Object.entries(m.horizons || {}).flatMap(([k, v]) => [h('dt', null, `${k} h`), h('dd', null, `MAE ${num(v.mae, 2)} min · n ${v.n}`)])));
    parts.push(h('h3', null, 'By weather zone'), h('dl', { class: 'kv' }, Object.entries(m.per_zone || {}).flatMap(([z, v]) => [h('dt', null, z), h('dd', null, typeof v === 'string' ? v : `MAE ${num(v.mae, 2)} min · n ${v.n}`)])));
    const c = m.conformal;
    parts.push(h('h3', null, '80% interval'), h('p', null, `Target ${pct(c.target)}; coverage on the test block ${pct(c.test_coverage, 1)} (n = ${c.test_n}), mean width ${num(c.mean_width_min, 1)} min. Calibration rows: ${c.n_cal}.`));
  }
  return h('div', { class: 'panel stack' }, parts);
}

export default async function mount(root) {
  const body = h('div', { class: 'stack' }, loading('loading model cards'));
  root.append(h('section', { class: 'page' },
    h('div', { class: 'hero' }, h('h1', null, 'Models and data'), h('span', { class: 'lede' }, 'What was trained on what, how it scores against simple baselines, and how fresh each data source is.')), body));
  try {
    const j = await api.models();
    const c = j.card;
    if (!j.trained) { put(clear(body), notice('Not trained', 'No trained models are installed. Run `python -m aerogrid.train.train_all` (see TODO.md).', 'warn')); return; }
    put(clear(body), 
      h('div', { class: 'row' }, trainedBadge(c.trained_on), h('span', { class: 'small mono' }, `built ${c.created_at}`)),
      c.trained_on === 'synthetic' ? notice('Synthetic', c.note, 'warn') : notice('Real data', c.note),
      reduced(c) ? notice('Reduced run', `Trained on only ${c.source_meta.months.length} month(s) (${c.source_meta.months[0]} to ${c.source_meta.months[c.source_meta.months.length - 1]}) of US data. The design trains on 30 months in GitHub Actions; with this little data most components beat their naive baselines only marginally. Read the baselines beside every number.`, 'warn') : null,
      notice('Reading the metrics', 'Delays are heavy-tailed: MAE rewards low predictions, RMSE rewards getting the average right, and RMSE is the headline. Both are shown, each with n and a 95% interval from a block bootstrap over test days. A model counts as better than a baseline only if the whole interval of its RMSE ratio is below 1; the test block is only a few days long, so most intervals are wide.'),
      h('div', { class: 'panel' }, h('h2', null, 'Training protocol'),
        h('dl', { class: 'kv' },
          h('dt', null, 'window'), h('dd', null, `${c.split_dates.start} to ${c.split_dates.end}`),
          h('dt', null, 'train until'), h('dd', null, c.split_dates.train_end), h('dt', null, 'calibration'), h('dd', null, c.split_dates.calib.join(' to ')), h('dt', null, 'test from'), h('dd', null, c.split_dates.test_start),
          h('dt', null, 'flights (train / calib / test)'), h('dd', null, `${c.counts.train_flights} / ${c.counts.calib_flights} / ${c.counts.test_flights}`), h('dt', null, 'airports in graph'), h('dd', null, c.counts.nodes)),
        h('p', { class: 'small', style: { marginTop: '8px' } }, 'Blocks are time-ordered with gaps, never shuffled; every statistic is fitted on train only; reported numbers use the test block only.'),
        c.source_meta && c.source_meta.limitations ? h('ul', { class: 'plain small adv' }, c.source_meta.limitations.map((l) => h('li', null, l))) : null,
        h('p', { class: 'small' }, `Inference graph: ${c.inference_graph}`)),
      h('div', { class: 'grid two' }, Object.entries(c.models).map(([k, m]) => card(k, m))),
      h('div', { class: 'panel' }, h('h2', null, 'Data sources'), h('div', { class: 'tscroll' }, h('table', { class: 'tbl' },
        h('thead', null, h('tr', null, ['Source', 'Role', 'Cache', 'Licence', 'Last call'].map((t) => h('th', null, t)))),
        h('tbody', null, j.data_sources.map((s) => h('tr', null, h('td', null, s.name, s.enabled === false ? h('span', { class: 'badge b-low', style: { marginLeft: '6px' } }, 'off') : null), h('td', null, s.role), h('td', { class: 'mono' }, `${s.ttl_s} s`), h('td', { class: 'small' }, s.license),
          h('td', null, s.last ? [badge(s.last.source, s.last), ' ', h('span', { class: 'small mute' }, ago(s.last.age_s))] : h('span', { class: 'mute small' }, 'not called yet')))))))),
      h('p', { class: 'small' }, `Scorecard ground truth: ${j.badges.ground_truth}.`),
    );
  } catch (e) { put(clear(body), errorBox(e)); }
}
