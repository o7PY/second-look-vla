// Part pages: one hands-on demo each. All numbers come from SITE (ledger rows, plus sourced facts).
(function () {
  const { S, V, CI, pct, hbars, ciSwitch, segmented, countUps, reveals, reduce } = SL;
  const $ = id => document.getElementById(id);
  const part = +document.body.dataset.part;
  reveals(); countUps(); ciSwitch($('ci'));
  const row = (label, id, cls, extra = {}) => ({ label, v: V(id), ci: CI(id), cls, ...extra });
  const plural = (n, w) => n + ' ' + w + (n === 1 ? '' : 's');

  // ---------------- Part 1: how well does it work ----------------
  if (part === 1) {
    // size the walkthrough frame to its content, so there is no scroll box inside the page
    const fr = document.querySelector('.frame iframe');
    if (fr) fr.addEventListener('load', () => {
      try {
        const d = fr.contentDocument, fit = () => { fr.style.height = Math.ceil(d.documentElement.scrollHeight) + 'px'; };
        fit(); new ResizeObserver(fit).observe(d.body); fr.contentWindow.addEventListener('resize', fit);
      } catch (e) { /* cross-origin (file://): keep the CSS height */ }
    });
    const F = S.F, mk = (label, v, cls) => ({ label, v: +v, ci: null, cls });
    hbars($('auroc'), [
      mk('Test set (3-run mean)', F['c.test_auroc'], 'hl'),
      mk('Tasks it never trained on (3-run mean)', F['c.heldout_auroc'], 'alt'),
      mk('SmolVLA episodes in simulation (3-run mean)', F['c.policy_auroc'], 'base'),
      mk(F['sealed.eps'] + ' sealed fresh episodes (3-run mean)', F['sealed.auroc_C'], 'base'),
      row('In the loop, seen tasks (seed 0)', 'p2.gate.auroc_seen', 'base'),
      row('In the loop, all tasks (seed 0)', 'p2.gate.auroc_all', 'base')
    ], { min: 0.5, max: 1, fmt: v => v.toFixed(3), caption: 'AUROC of model C by situation, bars start at 0.5 (a coin flip)', valHead: 'AUROC', rowHead: 'Situation' });
  }

  // ---------------- Part 2: compare the arms ----------------
  if (part === 2) {
    const ARMS = [
      ['P0', 'Do nothing', 'The robot simply carries on. This is the baseline everything is compared with.'],
      ['P1', 'Re-roll', 'A fresh attempt with new randomness, with no camera check.'],
      ['P2', 'Blind retry', 'Rewind the arm and retry whenever the alarm fires, without looking at the scene.'],
      ['P3', 'Second Look', 'Pause, look, then retry if only the arm moved. If something else moved, flag a human.'],
      ['P4', 'Oracle gate', 'Same as Second Look, but the simulator’s own truth replaces the camera. If this matches Second Look, the camera is not the bottleneck.'],
      ['P5', 'Random timing', 'The same kind of check, but triggered at random moments instead of by the alarm.'],
      ['P6', 'Check only', 'Pause and look, but never rewind.']
    ];
    const METRIC = { S: ['Succeeded', 'hl'], SF: ['Failed silently', 'alt'], F: ['Flagged to a human', 'base'] };
    let metric = 'S', sel = 3;
    const chart = $('arms'), info = $('arm-info');
    function describe(i) {
      const [k, name, d] = ARMS[i];
      const r = S.L[`p2.seen.${k}.rescue`], h = S.L[`p2.seen.${k}.harm`];
      info.innerHTML = `<b>${name}.</b> ${d}` + (r ? `<br><span class="mut">Compared with doing nothing it rescued <b>${r[0]}</b> runs that would have failed and hurt <b>${h[0]}</b> runs that would have succeeded.</span>` : '<br><span class="mut">This is the reference: rescues and harms are counted against it.</span>');
    }
    function draw() {
      const [t, c] = METRIC[metric];
      $('arms-ttl').textContent = t + ' (share of 720 runs on the eight seen tasks)';
      hbars(chart, ARMS.map(([k, name], i) => row(name, `p2.seen.${k}.${metric}`, i === sel ? c : 'base', {
        tip: `<b>${name}</b><br>${pct(V(`p2.seen.${k}.${metric}`))}<br><span class="mut">click for details</span>` })), {
        max: metric === 'S' ? 1 : 0.5, fmt: v => pct(v), caption: t, onPick: (r, i) => { sel = i; draw(); describe(i); } });
      chart.querySelectorAll('.hb-row')[sel].classList.add('sel');
    }
    segmented($('metric'), k => { metric = k; draw(); });
    describe(sel);
  }

  // ---------------- Part 3: retry or continue ----------------
  if (part === 3) {
    const BASE = [['P0', 'Do nothing'], ['PC', 'Always continue'], ['P2r', 'Always retry'], ['SH', 'Learned selector']];
    const EXTRA = [['SH-task', 'Task id only'], ['SH-noG', 'No camera score'], ['SH+F', 'Selector + flag']];
    let metric = 'S', more = false;
    function draw() {
      const arms = more ? BASE.concat(EXTRA) : BASE, hl = k => k === 'SH';
      const host = $('arms');
      if (metric === 'S') {
        $('arms-ttl').textContent = 'Succeeded (share of 1000 runs)';
        hbars(host, arms.map(([k, n]) => row(n, `p4.${k}.S`, hl(k) ? 'hl' : 'base')), { max: 1, caption: 'Success' });
      } else if (metric === 'R') {
        $('arms-ttl').textContent = 'How often the robot retried';
        hbars(host, arms.map(([k, n]) => ({ label: n, v: V(`p4.${k}.retry_rate`), cls: hl(k) ? 'hl' : 'base' })), { max: 1, caption: 'Retry rate' });
      } else {
        $('arms-ttl').textContent = 'Runs rescued and runs hurt, compared with doing nothing';
        const rows = [];
        arms.filter(([k]) => k !== 'P0').forEach(([k, n]) => { rows.push({ label: n + ' – rescued', v: V(`p4.${k}.rescue`), cls: 'hl' }); rows.push({ label: n + ' – hurt', v: V(`p4.${k}.harm`), cls: 'alt' }); });
        hbars(host, rows, { max: 70, fmt: v => String(Math.round(v)), caption: 'Rescues and harms', valHead: 'Runs', rowHead: 'Arm' });
      }
    }
    segmented($('metric'), m => { metric = m; draw(); });
    const sw = $('more');
    sw.addEventListener('click', () => { more = !more; sw.setAttribute('aria-pressed', more); draw(); });
  }

  // ---------------- Part 4: the noise ceiling and the critic ----------------
  if (part === 4) {
    const KS = [1, 2, 4, 8, 16, 32, 64], IDS = ['p5.0.mean_p', 'p5.0.any2', 'p5.0.any4', 'p5.0.any8', 'p5.0.any16', 'p5.0.any32', 'p5.0.any64'];
    const N = S.F['p5.states'], grid = $('dots'), slider = $('k');
    for (let i = 0; i < N; i++) grid.appendChild(document.createElement('i'));
    function setK() {
      const i = +slider.value, k = KS[i], v = V(IDS[i]), on = Math.round(v * N);
      $('k-o').textContent = k === 1 ? '1 draw (a plain retry)' : k + ' draws';
      $('k-v').textContent = pct(v);
      $('k-n').textContent = `about ${on} of ${N} failed situations`;
      grid.querySelectorAll('i').forEach((d, j) => d.classList.toggle('on', j < on));
      slider.setAttribute('aria-valuetext', `${k} draws: ${pct(v)} rescued`);
    }
    slider.addEventListener('input', setK); setK();

    // three kinds of situation (counts = share x 40)
    const cls = $('classes'), z = Math.round(V('p5.0.dist_zero') * N), m = Math.round(V('p5.0.dist_mid') * N), h = N - z - m;
    [['z', z], ['m', m], ['h', h]].forEach(([c, n]) => { for (let i = 0; i < n; i++) { const d = document.createElement('i'); d.className = c; cls.appendChild(d); } });

    // critic gauge on a 0.5 to 1.0 axis
    const pos = v => ((v - 0.5) / 0.5) * 100, g = $('gauge'), a = V('p5.1a.auc'), ci = CI('p5.1a.auc');
    g.innerHTML = `<div class="rail0"></div><div class="ci2" style="left:${pos(ci[0])}%;width:${pos(ci[1]) - pos(ci[0])}%"></div>
      <div class="mk dn l" style="left:0%"><span>chance 0.50</span></div>
      <div class="mk you up" style="left:${pos(a)}%"><span>our critic ${a.toFixed(3)}</span></div>
      <div class="mk dn" style="left:${pos(S.F['p5.bar_auc'])}%"><span>needed ${S.F['p5.bar_auc'].toFixed(2)}</span></div>
      <div class="mk up r" style="left:100%"><span>perfect 1.0</span></div>`;
    g.setAttribute('role', 'img'); g.setAttribute('aria-label', `A critic ranks good noise above bad with AUROC ${a.toFixed(3)}; chance is 0.5, the pre-registered bar was ${S.F['p5.bar_auc']}, perfect is 1.0`);
  }
})();
