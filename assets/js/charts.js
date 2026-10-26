// Small chart helpers (plain DOM, no libraries). Every number comes from window.SITE (generated from the ledger).
const S = window.SITE || { L: {}, F: {} };
const V = id => { if (!(id in S.L)) throw new Error('unknown ledger id ' + id); return S.L[id][0]; };
const CI = id => (S.L[id].length > 2 ? [S.L[id][1], S.L[id][2]] : null);
const pct = (v, d = 1) => (100 * v).toFixed(d) + '%';
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

// one shared tooltip
let tip;
function showTip(html, x, y) {
  if (!tip) { tip = document.createElement('div'); tip.className = 'tip'; tip.setAttribute('role', 'tooltip'); document.body.appendChild(tip); }
  tip.innerHTML = html; tip.style.opacity = 1;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(8, Math.min(innerWidth - w - 8, x - w / 2)) + 'px';
  tip.style.top = Math.max(8, y - h - 12) + 'px';
}
const hideTip = () => { if (tip) tip.style.opacity = 0; };

// Horizontal bars. rows: [{label, v, ci:[lo,hi]|null, cls:'base'|'hl'|'alt', tip:'html', note:'text'}]
// opts: {max, fmt(v), caption, onPick(row,i)}   v in [0, max]; bars grow from the baseline, data end rounded.
function hbars(host, rows, opts = {}) {
  const max = opts.max ?? 1, min = opts.min ?? 0, span = max - min, fmt = opts.fmt || (v => pct(v));
  host.classList.add('hb'); host.innerHTML = '';
  if (opts.caption) host.setAttribute('aria-label', opts.caption);
  rows.forEach((r, i) => {
    const row = document.createElement('div');
    row.className = 'hb-row' + (opts.onPick ? ' pick' : '');
    row.tabIndex = 0;
    const lab = `${r.label}: ${fmt(r.v)}${r.ci ? ` (95% CI ${fmt(r.ci[0])} to ${fmt(r.ci[1])})` : ''}`;
    row.setAttribute('aria-label', lab);
    const w = Math.max(0, Math.min(100, ((r.v - min) / span) * 100));
    const ci = r.ci ? `<span class="whisk" style="left:${Math.max(0, ((r.ci[0] - min) / span) * 100)}%;width:${((r.ci[1] - r.ci[0]) / span) * 100}%"></span>` : '';
    row.innerHTML = `<span class="hb-lab">${r.label}</span><span class="hb-track"><span class="hb-bar ${r.cls || 'base'}" data-w="${w}" style="width:${reduce ? w : 0}%"></span>${ci}</span><span class="hb-val">${fmt(r.v)}<span class="ci-txt"> ${r.ci ? '[' + fmt(r.ci[0]) + ', ' + fmt(r.ci[1]) + ']' : ''}</span></span>`;
    const tipHtml = r.tip || `<b>${r.label}</b><br>${fmt(r.v)}${r.ci ? `<br><span class="mut">95% CI ${fmt(r.ci[0])} to ${fmt(r.ci[1])}</span>` : ''}`;
    row.addEventListener('mousemove', e => showTip(tipHtml, e.clientX, e.clientY));
    row.addEventListener('mouseleave', hideTip);
    row.addEventListener('focus', () => { const b = row.getBoundingClientRect(); showTip(tipHtml, b.left + b.width / 2, b.top); });
    row.addEventListener('blur', hideTip);
    if (opts.onPick) { row.addEventListener('click', () => opts.onPick(r, i)); row.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opts.onPick(r, i); } }); }
    host.appendChild(row);
  });
  // grow on first view
  if (!reduce) {
    const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { en.target.querySelectorAll('.hb-bar').forEach(b => requestAnimationFrame(() => { b.style.width = b.dataset.w + '%'; })); io.unobserve(en.target); } }), { threshold: 0.3 });
    io.observe(host);
  }
  // table view of the same data (accessibility + copy/paste)
  let t = host.nextElementSibling;
  if (!(t && t.classList.contains('tv'))) { t = document.createElement('details'); t.className = 'tv'; host.after(t); }
  t.innerHTML = `<summary>View as table</summary><table><thead><tr><th>${opts.rowHead || 'Arm'}</th><th>${opts.valHead || 'Value'}</th><th>95% CI</th></tr></thead><tbody>${rows.map(r => `<tr><td>${r.label}</td><td>${fmt(r.v)}</td><td>${r.ci ? fmt(r.ci[0]) + ' to ' + fmt(r.ci[1]) : '-'}</td></tr>`).join('')}</tbody></table>`;
}

// "Show confidence intervals" switch: toggles body.show-ci (whiskers + bracketed text everywhere)
function ciSwitch(btn) {
  if (!btn) return;
  const set = on => { document.body.classList.toggle('show-ci', on); btn.setAttribute('aria-pressed', on); try { localStorage.setItem('sl-ci', on ? '1' : '0'); } catch (e) {} };
  let init = false; try { init = localStorage.getItem('sl-ci') === '1'; } catch (e) {}
  set(init);
  btn.addEventListener('click', () => set(!document.body.classList.contains('show-ci')));
}

// segmented control: buttons [data-k]; calls cb(k)
function segmented(host, cb) {
  const bs = [...host.querySelectorAll('button')];
  const pick = b => { bs.forEach(x => x.setAttribute('aria-pressed', x === b)); cb(b.dataset.k); };
  bs.forEach(b => b.addEventListener('click', () => pick(b)));
  pick(bs[0]);
}

// count-up when a number scrolls into view: any element with data-count (keeps the filled text as the final value)
function countUps(root = document) {
  root.querySelectorAll('[data-count]').forEach(el => {
    const final = el.textContent, m = final.match(/^([^\d−+-]*)([−+-]?)([\d,]*\.?\d+)(.*)$/);
    if (!m || reduce) return;
    const [, pre, sign, num, post] = m, dec = (num.split('.')[1] || '').length, target = parseFloat(num.replace(/,/g, '')), comma = num.includes(',');
    const fmtN = x => { const s = x.toFixed(dec); return comma ? Number(s).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) : s; };
    el.textContent = pre + sign + fmtN(0) + post;
    const io = new IntersectionObserver(es => es.forEach(en => {
      if (!en.isIntersecting) return; io.disconnect();
      const t0 = performance.now(), D = 1100;
      (function step(t) { const k = Math.min(1, (t - t0) / D), e = 1 - Math.pow(1 - k, 3); el.textContent = pre + sign + fmtN(target * e) + post; if (k < 1) requestAnimationFrame(step); else el.textContent = final; })(t0);
    }), { threshold: 0.6 });
    io.observe(el);
  });
}

// reveal-on-scroll for .rv elements
function reveals() {
  const els = document.querySelectorAll('.rv');
  if (reduce || !('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('in')); return; }
  const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } }), { threshold: 0.15 });
  els.forEach(e => io.observe(e));
}

window.SL = { S, V, CI, pct, hbars, ciSwitch, segmented, countUps, reveals, showTip, hideTip, reduce };
