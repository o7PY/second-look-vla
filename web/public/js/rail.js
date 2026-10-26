// Side rail: on the home page the lines scroll to the four part sections and follow the screen;
// on a part page they are links to the four part pages. Nothing is written next to the lines (aria-labels only).
(function () {
  const NAMES = ['Part 1: the detector', 'Part 2: the closed loop', 'Part 3: choosing when to retry', 'Part 4: steering the retry noise'];
  const rail = document.querySelector('.rail');
  if (!rail) return;
  const home = document.body.dataset.page === 'home';
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  rail.setAttribute('role', 'navigation');
  rail.setAttribute('aria-label', 'Parts');
  const items = NAMES.map((n, i) => {
    const el = document.createElement(home ? 'button' : 'a');
    el.setAttribute('aria-label', n);
    if (home) el.type = 'button'; else el.href = 'part' + (i + 1) + '.html';
    el.innerHTML = '<i></i>';
    rail.appendChild(el);
    return el;
  });
  const setCurrent = k => items.forEach((el, i) => el.setAttribute('aria-current', i === k ? 'true' : 'false'));
  if (!home) { setCurrent(+document.body.dataset.part - 1); return; }

  const secs = [1, 2, 3, 4].map(n => document.getElementById('p' + n));
  items.forEach((el, i) => el.addEventListener('click', () => secs[i].scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })));
  // current part = the section that covers the middle of the screen; the rail only shows from Part 1 on
  const parts = document.getElementById('parts');
  let cur = -1;
  function update() {
    const mid = innerHeight * 0.5;
    let k = -1;
    secs.forEach((s, i) => { const r = s.getBoundingClientRect(); if (r.top <= mid && r.bottom > mid) k = i; });
    if (k !== cur) { cur = k; setCurrent(k); }
    const top = parts.getBoundingClientRect().top;
    rail.dataset.hidden = top > innerHeight * 0.6 ? 'true' : 'false';
  }
  addEventListener('scroll', update, { passive: true });
  addEventListener('resize', update);
  update();
})();
