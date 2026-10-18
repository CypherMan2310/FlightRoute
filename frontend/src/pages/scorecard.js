// Scorecard: the app's own live accuracy against FAA NAS delay programmes (US airports only).
import { api } from '../api.js';
import { h, clear, trainedBadge, fmt, pct, errorBox, loading, notice, put } from '../ui.js';

const ci = (c) => (c ? `${pct(c[0])} to ${pct(c[1])}` : '–');

export default async function mount(root) {
  const body = h('div', { class: 'stack' }, loading('loading scorecard'));
  root.append(h('section', { class: 'page' },
    h('div', { class: 'hero' }, h('h1', null, 'Scorecard'), h('span', { class: 'lede' }, 'The app marks its own homework: predicted airport delays are logged, then compared with what the FAA actually reported.')),
    body));
  try {
    const m = await api.scorecard();
    const c = m.confusion;
    const base = m.baselines;
    put(clear(body), 
      h('div', { class: 'row' }, trainedBadge(m.trained_on), h('span', { class: 'small' }, `model horizon ${m.model_horizon_h} h · ${m.scored_airports} US airports scored`)),
      notice('Ground truth', m.ground_truth),
      notice('Event', m.event_definition),
      m.insufficient_data ? notice('Not enough data yet', `Only ${m.n_resolved} predictions have been resolved (30 are needed before the rates mean anything). ${m.n_pending} are waiting for their target hour; ${m.n_expired} expired unresolved.`, 'warn') : null,
      h('div', { class: 'grid three' },
        h('div', { class: 'panel stat' }, h('span', { class: 'k' }, 'Precision'), h('span', { class: 'v' }, pct(m.precision)), h('span', { class: 'small' }, `95% CI ${ci(m.precision_ci95)}`)),
        h('div', { class: 'panel stat' }, h('span', { class: 'k' }, 'Recall'), h('span', { class: 'v' }, pct(m.recall)), h('span', { class: 'small' }, `95% CI ${ci(m.recall_ci95)}`)),
        h('div', { class: 'panel stat' }, h('span', { class: 'k' }, 'F1'), h('span', { class: 'v' }, m.f1 === null ? '–' : fmt(m.f1, 2)), h('span', { class: 'small' }, `n = ${m.n_resolved} resolved`))),
      h('div', { class: 'grid two' },
        h('div', { class: 'panel' }, h('h2', null, 'Confusion matrix'), h('table', { class: 'tbl' },
          h('thead', null, h('tr', null, h('th', null, ''), h('th', { class: 'r' }, 'FAA: programme'), h('th', { class: 'r' }, 'FAA: normal'))),
          h('tbody', null, h('tr', null, h('td', null, 'Predicted event'), h('td', { class: 'r mono' }, c.tp), h('td', { class: 'r mono' }, c.fp)), h('tr', null, h('td', null, 'Predicted normal'), h('td', { class: 'r mono' }, c.fn), h('td', { class: 'r mono' }, c.tn))))),
        h('div', { class: 'panel' }, h('h2', null, 'Baselines'), h('table', { class: 'tbl' },
          h('thead', null, h('tr', null, ['', 'Accuracy', 'Recall', 'Precision'].map((t, i) => h('th', { class: i ? 'r' : '' }, t)))),
          h('tbody', null,
            row('This model', m.accuracy, m.recall, m.precision),
            row('Always "normal"', base.always_normal.accuracy, base.always_normal.recall, base.always_normal.precision),
            row('Persistence (FAA status at issue time)', base.persistence.accuracy, base.persistence.recall, base.persistence.precision))),
          h('p', { class: 'small' }, `Base rate of FAA programmes in the resolved sample: ${pct(m.base_rate)}.`))),
      h('div', { class: 'panel' }, h('h2', null, 'Minutes'), h('p', null, m.minutes_error.n ? `Mean absolute error on real events: ${fmt(m.minutes_error.mae, 1)} min (n = ${m.minutes_error.n}).` : 'No resolved events with a reported delay yet.')),
      h('p', { class: 'small' }, m.note),
    );
  } catch (e) { put(clear(body), errorBox(e)); }
}

const row = (name, acc, rec, prec) => h('tr', null, h('td', null, name), h('td', { class: 'r mono' }, pct(acc)), h('td', { class: 'r mono' }, pct(rec)), h('td', { class: 'r mono' }, pct(prec)));
