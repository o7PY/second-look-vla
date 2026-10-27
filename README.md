# Second Look: the website

**Before a robot retries, it takes a second look.**

This repository holds the explainer website for *Second Look*, a study of what a robot should do when its AI policy starts to fail. The site tells the story in plain language, with a scrolling home page and a page for each experiment, and lets you try the ideas hands-on.

## What the site is about

A robot driven by an AI policy sometimes fails halfway through a job. The obvious fix is to try again, but if the failure moved something, trying again from a changed scene can make things worse. Second Look asks whether the robot can notice trouble early, check what changed using only a camera, and then pick the right response: retry, carry on, or call a human. Everything was studied in simulation, with one policy (SmolVLA) on one benchmark (LIBERO-Goal).

## What you will find on it

- **Home page.** One scrolling story: the project and its aim, the setup (alarm, pause, look, decide, and the benchmark), the data that was generated, a summary of each experiment, and what was found. Four thin lines on the left edge jump between the experiments; the current one is the longest.
- **Part 1, the detector.** DeltaDINO, a change detector that compares a before photo and an after photo and says whether anything besides the robot arm moved. Includes a step-by-step walkthrough on one real image pair.
- **Part 2, the closed loop.** Pausing, looking, then retrying or flagging a human: an explorer for seven ways of handling alarms, and the three pre-registered tests.
- **Part 3, choosing when to retry.** A small learned model that decides between retrying and carrying on, compared with simple rules.
- **Part 4, steering the retry noise.** Whether some retries are luckier than others, how much room that leaves, and a critic that tries, and mostly fails, to pick the good ones.

The results are reported plainly: most of the pre-registered hypotheses were not supported, and the site says so.

## How the site is built

Plain HTML, CSS and JavaScript, with no framework. The home page has a dark, technical look; the experiment pages are light, with a dark mode that follows the system setting. It works on phones, respects reduced-motion settings, and can be used with a keyboard. Every number on the pages is read from the project's results ledger or from a sourced list of facts; none is typed by hand. Illustrations that are not real model output are labelled as such.

## What is in this repository

- `index.html`, `part1.html` to `part4.html`, `assets/`: the website itself, meaning the home page, the four experiment pages, and their styles, scripts and data.
- `.github/workflows/pages.yml`: an optional GitHub Actions workflow for publishing the website (run by hand).
- `web/`: the small tool that fills the site's numbers from the study's results.
- `src/`, `scripts/`, `tests/`: the Python code behind the study (the detector, the closed loop, the analyses and their tests).
