// Home page behaviour. Numbers come from SITE (ledger + sourced facts); illustrations are labelled as such in the page.
(function () {
  const { S, V, pct, hbars, countUps, reveals, showTip, hideTip, reduce } = SL;
  reveals(); countUps();

  // --- the loop: four steps, auto-cycling while visible, click to pin ---
  const STEPS = [
    'The robot works as normal. A watcher listens to how steady the policy’s own plan is. When the plan suddenly gets shaky, an alarm fires: something is probably going wrong.',
    `The robot freezes for ${S.F['bench.pause_steps']} steps (half a second) so everything settles, and takes a photo of the scene.`,
    'DeltaDINO compares the photo from before the trouble with the photo now. The question is only: did anything besides the robot arm move?',
    'Only the arm moved: rewind the arm a little and retry. Something else moved: stop and flag a human, because a retry could make it worse.'
  ];
  const steps = [...document.querySelectorAll('.step')], note = document.getElementById('loop-note');
  let k = 0, pinned = false, timer;
  const show = i => { k = i; steps.forEach((s, j) => s.setAttribute('aria-pressed', j === i)); note.textContent = STEPS[i]; };
  steps.forEach((s, i) => s.addEventListener('click', () => { pinned = true; clearInterval(timer); show(i); }));
  show(0);
  const loopEl = document.querySelector('.loop');
  if (!reduce) new IntersectionObserver(es => es.forEach(en => {
    clearInterval(timer);
    if (en.isIntersecting && !pinned) timer = setInterval(() => show((k + 1) % 4), 3200);
  }), { threshold: 0.4 }).observe(loopEl);

  // --- label rule demo: how far did an object move? ---
  const T = S.F, tr = document.getElementById('tr'), ro = document.getElementById('ro'), out = document.getElementById('verdict');
  function verdict() {
    const t = +tr.value, r = +ro.value;
    document.getElementById('tr-o').textContent = t.toFixed(1) + ' cm';
    document.getElementById('ro-o').textContent = r.toFixed(1) + '°';
    let v, h, s;
    if (t > T['thr.changed_cm'] || r > T['thr.changed_deg']) { v = 'changed'; h = 'Changed'; s = 'Counts as a real change: the detector should say yes.'; }
    else if (t < T['thr.same_cm'] && r < T['thr.same_deg']) { v = 'same'; h = 'Unchanged'; s = 'Too small to matter: the detector should say no.'; }
    else { v = 'gray'; h = 'Gray zone'; s = 'In between, so these pairs are left out of training and scoring.'; }
    out.dataset.v = v; out.innerHTML = h + '<small>' + s + '</small>';
  }
  tr.addEventListener('input', verdict); ro.addEventListener('input', verdict); verdict();

  // --- part 1 mini viz: an illustrative change map (a hot blob on a quiet grid) ---
  const cm = document.getElementById('cmap');
  if (cm) {
    const cells = []; for (let i = 0; i < 256; i++) { const c = document.createElement('i'); cm.appendChild(c); cells.push(c); }
    const cx = 5, cy = 9;
    const heat = (r, c) => Math.exp(-((r - cy) ** 2 + (c - cx) ** 2) / 4.2);
    function paint(on) { cells.forEach((c, i) => { const h = on ? heat(Math.floor(i / 16), i % 16) : 0; c.className = h > 0.75 ? 'hh' : h > 0.3 ? 'h' : ''; }); }
    paint(reduce);
    new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { setTimeout(() => paint(true), 300); } }), { threshold: 0.5 }).observe(cm);
  }

  // --- parts 2-3 mini bars (ledger values) ---
  const R = (label, id, cls) => ({ label, v: V(id), ci: SL.CI(id), cls });
  const mini = (id, rows, caption) => { const h = document.getElementById(id); if (h) hbars(h, rows, { caption }); };
  mini('p2-s', [R('Do nothing', 'p2.seen.P0.S', 'base'), R('Second Look', 'p2.seen.P3.S', 'hl')], 'Success');
  mini('p2-sf', [R('Do nothing', 'p2.seen.P0.SF', 'base'), R('Second Look', 'p2.seen.P3.SF', 'hl')], 'Silent failures');
  mini('p3-s', [R('Do nothing', 'p4.P0.S', 'base'), R('Always retry', 'p4.P2r.S', 'base'), R('Learned selector', 'p4.SH.S', 'hl')], 'Success');
  mini('p3-r', [R('Always retry', 'p4.P2r.retry_rate', 'base'), R('Learned selector', 'p4.SH.retry_rate', 'hl')], 'How often it retries');

  // --- part 4 mini viz: the any-of-k curve ---
  const cv = document.getElementById('p4-curve');
  if (cv) {
    const ks = [1, 2, 4, 8, 16, 32, 64], ids = ['p5.0.mean_p', 'p5.0.any2', 'p5.0.any4', 'p5.0.any8', 'p5.0.any16', 'p5.0.any32', 'p5.0.any64'];
    const ys = ids.map(V), W = 320, H = 190, L = 34, Rr = 14, Tp = 16, B = 34, ymax = 0.7;
    const X = i => L + (i / 6) * (W - L - Rr), Y = v => Tp + (1 - v / ymax) * (H - Tp - B);
    const d = ys.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
    cv.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="curve" role="img" aria-label="Share of failures rescued by the best of k noise draws, from ${pct(ys[0])} at one draw to ${pct(ys[6])} at 64">
      ${[0, 0.2, 0.4, 0.6].map(g => `<line x1="${L}" x2="${W - Rr}" y1="${Y(g)}" y2="${Y(g)}"/><text x="${L - 6}" y="${Y(g) + 4}" text-anchor="end">${Math.round(g * 100)}%</text>`).join('')}
      ${ks.map((k, i) => `<text x="${X(i)}" y="${H - 14}" text-anchor="middle">${k}</text>`).join('')}
      <text x="${(L + W - Rr) / 2}" y="${H - 1}" text-anchor="middle">noise draws tried</text>
      <path class="ln" d="${d}"/>
      <circle cx="${X(0)}" cy="${Y(ys[0])}" r="4"/><circle cx="${X(6)}" cy="${Y(ys[6])}" r="4"/>
      <text class="hi" x="${X(0) + 2}" y="${Y(ys[0]) + 20}">${pct(ys[0])}</text><text class="hi" x="${X(6) - 6}" y="${Y(ys[6]) - 10}" text-anchor="end">${pct(ys[6])}</text></svg>`;
    const svg = cv.querySelector('svg'), p = svg.querySelector('path.ln'), len = p.getTotalLength();
    svg.style.setProperty('--len', len);
    if (reduce) svg.classList.add('on');
    else new IntersectionObserver((es, io) => es.forEach(en => { if (en.isIntersecting) { svg.classList.add('on'); io.disconnect(); } }), { threshold: 0.5 }).observe(svg);
  }
})();
