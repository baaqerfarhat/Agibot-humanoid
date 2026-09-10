# Literature — selecting a correction configuration. Claude track.

2026-09-08. Every citation below was retrieved and checked; nothing is cited from memory. The second
track is searching independently.

## The problem

The correct configuration — acting subspace and estimator jointly — differs by cell, and choosing
wrongly is catastrophic in either direction: `11 rescued / 0 broken` at one cell and
`1 rescued / 15 broken` at another, from the same change.

## Four families, and what each actually offers

### 1. Multiple-model adaptive control — the canonical answer

**Narendra & Balakrishnan, "Adaptive control using multiple models," IEEE Transactions on Automatic
Control 42(2):171–187, February 1997.** A bank of candidate models runs in parallel; a switching
rule picks the one whose prediction error is smallest, optionally with tuning. This is precisely the
"which configuration" machinery, and it is mature.

**The obstacle here is severe, and it is this project's own central finding.** MMAC switches on
*prediction error*. This project has repeatedly measured that prediction error and task outcome are
**dissociated**: across 168 same-state probes the weighted correction improved the pooled scaled
physical-error ratio to 0.905 while worsening 92 individual probes, with one case where the model
objective fell to 10.5% of zero while physical error rose 2.69-fold. A selector driven by the
quantity we have measured to be uninformative about the outcome is unlikely to select correctly.

### 2. Unfalsified control — the property this problem actually needs

**Safonov & Tsao, "The unfalsified control concept and learning," IEEE Transactions on Automatic
Control 42(6):843–847, June 1997, doi:10.1109/9.587340.** A supervisor monitors plant data and
falsifies candidate controllers **without a plant model**. Its distinctive property:

> a controller need not be in the loop to be falsified; candidates can be falsified with open-loop
> data, or with data acquired while a *different* controller was in the loop.

**That is exactly the operation this problem requires** — deciding whether the six-input
configuration would help at a cell, using data recorded under the translation configuration, without
running it. No other family offers this directly.

**The obstacle.** Unfalsified control falsifies against a **cost functional of (u, y)**. Our
objective is binary task success, which is not such a functional — a grasp either closes on the
object or does not, and no L2 criterion on commands and motions is known here to track it. Adapting
the framework would require exhibiting a surrogate cost demonstrably monotone in task success, which
this project has spent the session failing to find.

### 3. Active fault-tolerant control with reconfiguration

**Zhang & Jiang, "Bibliographical review on reconfigurable fault-tolerant control systems," Annual
Reviews in Control 32(2):229–252, 2008.** The standard architecture is exactly ours: fault detection
and diagnosis feeding a reconfiguration mechanism that selects the controller. So the problem is a
recognised one with a named architecture, which is worth stating in a paper.

The literature's reconfiguration decisions are keyed to *identified fault parameters*. Here the
fault is identified (`M⁻¹r` recovers it) and the configuration decision still goes wrong, so fault
identity is not the missing input. That is a genuine gap rather than an application failure.

### 4. Control allocation under uncertain effectiveness

**Johansen & Fossen, "Control allocation — A survey," Automatica 49(5), 2013.** Directly relevant to
the subspace half. Notably, recent work claims **adaptive control allocation that does not require
uncertainty estimation or persistency of excitation** — which matters because G5 concluded `E` is
not identifiable from a frozen policy's excitation. If allocation can be made robust without
identifying `E`, that route around the identifiability obstacle exists in the literature.

### 5. The closest contemporary work, and it is in our exact setting

**Ghaleb, Allahloh, Mejjaouli, Ali & Al-Shayea, "Uncertainty-Calibrated Safety Gating for
Vision–Language–Action Manipulation Under Domain Shift," Sensors 26(10), May 2026.** A calibrated
failure probability `r_t = 1 − p_succ(o_t, w)`, temperature-scaled and computed **before** acting,
gates between executing the policy, pausing to re-observe, and a fallback planner, with hysteresis
thresholds 0.2/0.5. Reports 57.5% → 77.2% under shift, calibration error 0.303 → 0.100.

**Why it matters here:** its gating signal predicts **task failure**, not model error. That is
exactly the axis on which the classical families fail for this problem. Its limitations are stated
honestly — simulation-only, no formal guarantees, residual collisions in the pause and fallback
states.

**Why it is not the answer as-is:** it gates *whether* to act, choosing among policy / pause /
fallback. It does not choose among **correction configurations**, which is our question.

## Assessment

**The classical machinery exists and is mature, and its selection signal is the one this project has
measured to be uninformative.** That is the honest summary. MMAC, unfalsified control and FTC
reconfiguration all key on prediction or tracking error; the session's evidence is that better
estimates coincide with worse task outcomes here.

The one contemporary line that keys on *predicted task failure* is in our setting but answers a
different question.

**So the gap is real and specific:** a selector over correction configurations, keyed to predicted
task outcome rather than model error, for a frozen policy. The literature supplies the architecture
(Zhang & Jiang), the data-reuse property that makes it cheap (Safonov & Tsao), and a task-failure
signal that works in this setting (Ghaleb et al.) — but no one has combined them for configuration
selection.

## The experiment this suggests

Unfalsified control's data-reuse property is the cheapest thing to test, and it is testable on
stored data: **from telemetry recorded under the translation configuration alone, can any statistic
distinguish the cell where six inputs help from the cell where they destroy?** Both cells' telemetry
already exists. If no statistic separates them, the selection problem is harder than the literature's
machinery assumes, and that is itself the finding.
