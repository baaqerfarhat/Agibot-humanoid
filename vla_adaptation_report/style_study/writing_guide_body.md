# What to learn from ICLR oral and award papers, and how to apply it here

The most useful lesson is **a clear scientific dependency, supported by the right evidence**. A strong paper explains what was previously possible, the precise obstacle or unexplained observation, the idea that changes the situation, and the experiment or argument that could establish that change. Attractive figures and compact prose make that dependency easier to inspect. They cannot compensate for a missing control, an unsupported novelty claim, or a theory whose assumptions do not cover the implementation.

This guide combines whole-corpus structural measurements with the targeted paper readings collected below. It is an observational study of selected successful papers, without a rejected-paper control group; it cannot establish that a particular style causes acceptance or an award. Recommendations labeled as editorial choices are our synthesis, not ICLR requirements.

## 1. Choose the story that the evidence supports

Different contributions call for different narratives. The close readings show at least four useful forms:

| Contribution | Narrative dependency | Examples in the collected papers | Main risk |
|---|---|---|---|
| New capability | A concrete resource or interface barrier → a construction that removes it → evidence of the resulting capability | DreamFusion; Visual Token Matching; SAM 2; Learning Interactive Real-World Simulators | Attractive examples without a sufficiently controlled evaluation |
| Mechanistic finding | A surprising observation → competing explanations → discriminating intervention → scope of the explanation | Blind Navigation; Vision Transformers Need Registers; Safety Alignment; Learning Dynamics | Treating a correlation or intermediate metric as a causal explanation |
| Unification or capability criterion | Existing categories miss a shared structure → a precise criterion → limits and constructive results → tests of the criterion | Graph Biconnectivity; SSL duality; Transformers are Inherently Succinct | Calling standard mathematics new, or implying broader equivalence than proved |
| Efficient implementation | A defined bottleneck → the computational structure that removes it → implementation details → quality/cost comparisons | Data Shapley; Faster Cascades; The Polar Express; Differentiable MPC | Mixing a new objective with a faster implementation of the old objective |

The year-specific profiles below link every example to its primary PDF and exact page/figure locations. These categories are analytical groupings, not official conference tracks.

**The appropriate story for the current adaptation paper is a mechanistic investigation with explicit local conditions and controlled interventions.** The evidence supports a useful distinction between residual information, correction authority, and feedback stability. It does not support “a new composite law is the best observer” or “physical faults are now solved.” The strong narrative is that a plausible design objective—identify the disturbance and correct it—has separable requirements that become visible when the same interface is tested beyond a favorable fault.

The contribution can be significant without a universally winning new estimator. It must nevertheless supply something beyond a list of failed trials: an explicit interface formulation, usable conditional statements, counterexamples that identify distinct architectural failure modes, and independent experiments that test their practical relevance. Standard linear algebra and adaptive-control tools should be credited as such. The novelty claim belongs to the specific formulation, connections, implementation analysis, and evidence, subject to comparison with the closest prior work.

## 2. Write an introduction that earns the importance claim

An introduction is not a compressed survey. Its paragraphs should change the reader's understanding in a deliberate order. A practical five-part structure for this paper is:

1. **State the concrete dependency.** A VLA's action must still produce the expected physical motion. Explain the deployment change at that interface and why adapting there is useful. Name the relevant policy lineage with primary citations; avoid a generic opening about AI transforming every industry.
2. **Show the tension.** The adapter repairs some command offsets yet degrades a joint-torque condition. Identify the different protocols explicitly. The observation motivates a question; it does not already prove a mechanism.
3. **Name the missing distinction.** The residual's information, the available correction directions, and the resulting closed-loop dynamics are separate objects. Define each in one sentence and give one concrete way it can fail.
4. **Explain the proposed intellectual move.** A common interface formulation permits local cancellation analysis, mismatch bounds, and a discrete composite feedback analysis. Explain why these are the right objects before listing implementation modules.
5. **State contributions as verifiable deliverables.** Each item should identify an object, a result, and where it is tested or proved. Distinguish conditions, an implemented variant, and independently confirmed outcomes.

This mirrors the useful question chain in [Blind Navigation](https://arxiv.org/pdf/2301.13261v1) and the criterion-first structure of [Graph Biconnectivity](https://arxiv.org/pdf/2301.09505v3), while keeping this paper's own scientific claim narrower. The aim is a reader who can explain the paper's question before reaching the method.

**How to sound current and fundamental:** identify a currently important system and a persistent underlying problem. Here the current system is a frozen VLA; the underlying problem is the gap between identifying a disturbance and correcting its effect through a restricted, imperfectly calibrated interface. Importance comes from that dependency and its consequences. Words such as “fundamental,” “paradigm,” “first,” and “universal” do not establish it. A “first” claim requires a defensible literature search and an exact scope; there is no need to make one here.

**A useful opening sentence for this draft:** “A robot can identify a disturbance accurately and still fail to compensate for it.” The next sentences must then define the setting, explain the gap, and say what evidence the paper supplies. This sentence is a general possibility illustrated by the conditional analysis; it is not a claim that every failed episode contains an accurate disturbance estimate.

**Abstract blueprint:** problem and setting; central distinction; principal technical results; experiment scope with development versus confirmation separated; one favorable result and the important boundary. Give the comparison denominator and comparator for any headline number. “15/60 to 41/60” is weighted versus legacy on the registered joint-5 comparison, not weighted versus an undisturbed robot. State the unresolved composite–Kalman result plainly.

## 3. Make related work a map of decisions

Dense related work compresses **relationships**, not merely names. Each paragraph should answer a question needed to locate the new contribution. A useful unit is: organizing claim → two or more relevant approaches → precise difference or shared limitation → consequence for this paper. A long paragraph with ten citations but no comparison is bibliographically dense and scientifically sparse.

For this draft, four organizing questions work well:

| Question | Literature to place here | Difference the reader must understand |
|---|---|---|
| Where does adaptation enter the robot stack? | RT-2, OpenVLA/OFT, Diffusion Policy, Octo, π0/π0.5 | Learning the observation-to-action policy versus adapting its command-to-motion interface |
| What is learned, and what receives the correction? | Adaptive augmentation with L1 control, DATT, residual policy learning, RMA | Adaptive controller, learned residual action, or learned context; training information and available action coordinates |
| What has already been shown about physical VLA faults? | J-PARC and directly relevant fault evaluations | Closest prior evidence about local correction versus task repair; healthy identification versus faulted development and reference information |
| Which objective guides adaptation? | DOB, analytical redundancy, robust and composite adaptive control; Neural-Fly, HMAC, MAGIC-VFM, control-oriented meta-learning | Prediction, tracking, learned disturbance representation, adaptation geometry, and what is actually held fixed in our comparisons |

The manuscript retains primary references for these claims, including the previously verified published MAGIC-VFM and HMAC records. The new writing study does not introduce unverified citation metadata or insert unrelated award papers into the robot-control bibliography.

For a closest-method comparison, inspect five axes before writing “unlike”: **observations; training data; online updated quantity; correction interface; claimed guarantee/evaluation**. For example, our plant models use healthy data, but mask and hyperparameter selection use faulted development outcomes. Omitting the second fact would manufacture a stronger distinction from prior work than the experiment supports.

Section placement is flexible. Some sampled papers put related work second, others after the method or near the end; several expand it in the appendix. The invariant is that readers encounter the nearest alternative before they are asked to evaluate the claimed novelty. Because this draft draws from both VLA and adaptive control, a compact main-text map is valuable. A supplementary literature table can expand it, but the nearest competing paper cannot be hidden there.

## 4. Explain the method from the executed computation outward

The reader should be able to reconstruct one control step from the main paper. A clear sequence is:

1. **Setting and available information.** State observations, action coordinates, sampling rate, fault insertion points, and which quantities are unavailable online. Separate command offsets from torque disturbances below the low-level controller.
2. **Offline fitting.** State what healthy data estimate and which models remain fixed. Separate model fitting from faulted development-time selection. Give the data split in the experiment section and the exact configuration in the appendix.
3. **Online signal path.** Define the raw policy action, applied correction, command sent to the controller, prediction, residual, observer update, and next-step timing in that order.
4. **One central equation per role.** The residual equation explains what is measured; the estimator explains what is inferred; the allocator explains what is applied. Reuse notation in the flowchart.
5. **What the variant changes.** Annotate the prediction-correction and tracking-correction terms of the composite update. State the zero-tracking/zero-damping reduction to Kalman with identical remaining parameters. An ablation knob is part of the method's clarity.
6. **Operational qualifications.** State gate semantics, target normalization, projection, the reference's use of raw commands, and the qualification limits that affect the argument. Move lengthy recursions and numerical implementation details to the appendix, not the definition of the executed method.

[DreamFusion's gradient annotations](https://arxiv.org/pdf/2209.14988v1) and [AlphaEdit's adjacent updates](https://proceedings.iclr.cc/paper_files/paper/2025/file/29c8c615b3187ee995029284702d3f43-Paper-Conference.pdf) are useful presentation examples. [Faster Cascades](https://proceedings.iclr.cc/paper_files/paper/2025/file/6f43166f50f26e8d8f3edc5545b0749f-Paper-Conference.pdf) gives an especially useful model for separating what a rule decides from how it is executed.

**Avoid a method-name contest.** Proposed/DOB/RLS/Kalman/calibrated-integral/oracle share some components but differ in update semantics and information. Compare residual construction, update rule, projection, permitted inputs, and privileged information independently. A raw-error integral baseline changes the signal as well as the update. Weighted-full changes support and regularization. Independently selected composite and Kalman settings change more than tracking feedback. These facts belong near the comparisons.

**Notation checklist:** define every symbol at first use; name the dimensions when they resolve ambiguity; distinguish the calibrated map from the unknown physical response; distinguish raw and corrected commands; use one index convention for action and observation timing; label which quantities are estimates, fixed fits, or idealized theoretical objects. Do not use a single symbol for a nominal calibration and a globally valid physical Jacobian without qualification.

## 5. Make theory strengthen the claim without changing its scope

A theorem should answer a question that the method creates. Useful main-text structure is: reader question → assumptions and mathematical object → statement → one-sentence interpretation → proof idea or small example → operational consequence. Full proofs belong in the appendix, but the assumptions that make the statement true belong beside it.

For this paper, the sequence is coherent:

| Question | Main statement | What it establishes | What remains separate |
|---|---|---|---|
| Does the residual contain what is needed, and do the inputs have the needed directions? | Kernel/range cancellation conditions | Exact cancellation in the stated noiseless local linear model | Bounded feasibility, nonlinear trajectories, and task success |
| Can a good full inverse become a bad restricted correction? | Inverse-then-mask counterexample | A concrete architectural failure even with an exact map | Whether this is the cause of a particular robot failure |
| Does a smaller fitted objective improve the real motion? | Physical-error bound with calibration/proxy uncertainty | A sufficient improvement condition if the uncertainty bounds hold | Uniform uncertainty bounds are not certified in the experiment |
| Does adding position feedback preserve stability? | Actual discrete augmented recursion and common-metric bound | Conditional stability/robustness of that model and timing | Healthy-model contraction alone and individual frozen-gain tests are insufficient |
| Does tracking imply task repair? | Explicit additional trajectory robustness requirement | What a transfer argument would need | A position RMS statistic alone does not supply the requirement |

The small geometry and scalar stability plots carry intuition, as in [Biological Constraints](https://arxiv.org/pdf/2210.01768v2). The selected 2026 theory/control examples also keep a short proof mechanism or annotated bound in the main text. Exact algebra should be distinguished from finite numerical implementation and empirical extrapolation. Do not relabel a triangle-inequality bound or a standard stability argument as a new general control theory.

For composite adaptation specifically, lower prediction or tracking error is a plausible design objective, not a proof of a best-estimator ranking. A useful new theory would explain conditions under which a feedback term helps a specified closed-loop quantity and quantify the approximation terms that can reverse that benefit. It should then motivate a controlled experiment that changes that term while holding other parameters fixed.

## 6. Design the experimental section around claims and alternatives

Lead each subsection with the question or finding, then provide enough protocol to assess it, then the result and its interpretation. Avoid chronological lab notes. Search, validation, and confirmation are different evidence stages and should remain separate even when all runs used the same simulator.

An experiment-to-claim map for this paper is:

| Claim or uncertainty | Needed comparison | Main display | Interpretation boundary |
|---|---|---|---|
| Calibrated residuals can support command repair | Faulted off versus calibrated correction, with protocol and calibration provenance | Historical counts and selected behavioral frames | Historical overlap prevents calling all calibration held out |
| Repair may depend on joint and task | Every allocated joint/suite plus healthy controls | Full condition matrix and representative primary/harm table | Equal torque is not equal fractional actuator severity |
| Allocation affects the available correction | Legacy versus frozen weighted-full on new scenarios | Registered joint-5 contrast, plus all secondary cells | Support and regularization change together |
| Nominal fit differs from physical response | Controlled local interventions in the same metric | Probe results and theory illustration | A local intervention does not identify every task failure |
| Composite beats a tuned alternative | Frozen family selection; untouched paired confirmation seeds | Full method/control table plus paired difference interval | A nonsignificant difference establishes neither superiority nor equivalence |
| Tracking feedback itself helps | Identical observer settings, tracking/damping factors varied | A future matched-parameter ablation | Current selected-controller diagnostics are confounded by other settings |

**Essential main-text settings:** robot and task; backbone/checkpoint family; control coordinates and rate; fault type, magnitude, duration and location; correction support and bounds; experimental unit and seeds; fitting/selection/confirmation split; baseline information; metric and aggregation; pairing/randomness; primary comparison and multiplicity. If changing a setting would change the interpretation of the headline, it belongs in the main paper or its immediately adjacent caption.

**Statistical presentation:** show counts and denominators for success; name the independent unit; report effect size and uncertainty for the primary comparison; distinguish paired outcomes from identical-policy-randomness counterfactuals; specify the correction family for adjusted p-values. An aggregate can conceal both rescues and regressions, so retain complete per-condition outcomes. Do not bold a method as the statistical winner merely because its point estimate is largest.

**Ablations must isolate a proposed mechanism.** If the method has three claimed ingredients, each needs either a controlled ablation or a clear statement that its individual contribution remains unidentified. More benchmark episodes cannot by themselves separate simultaneous changes in the observer, calibration, channel selection and allocator.

## 7. Give every figure one principal job

The close readings do not support a universal figure count or a mandatory first-page teaser. Theoretical papers can be effective with no figures; a systems paper may need a task picture, architecture, protocol diagram and several outcome displays. For this draft, four main figures have distinct roles:

| Figure | Reader question | Recommended content and current implementation |
|---|---|---|
| 1: interface and contrasting outcomes | What is adapted, where do faults enter, and why is the question real? | Frozen policy icon, signed command path, FIR/residual/observer/allocator loop, optional composite branch, separate command/torque injections; authentic Panda and ALOHA frames; measured success-rate decline panel |
| 2: analytical examples | Why can apparently reasonable correction fail? | Equal-scale geometric authority example and a scalar composite stability curve; label both as constructed models |
| 3: complete confirmation outcomes | Where does repair help or hurt, and is the tuned comparison resolved? | All Panda joint/suite cells versus shared off controls; all estimator/control scores; primary paired difference interval |
| 4: diagnostic dissociation | Does better reference tracking explain greater task success? | Tracking across conditions and known-offset parameter error; define the pooled metric and disclose differing selected parameters |

Figure 1 should mainly use symbols and imagery, with a short legend and caption defining the operators. Arrows must mean a specific information or action dependency. Feedback timing, raw versus corrected commands, and the fault insertion points must be visible. A decorative robot photograph that is disconnected from the method adds little.

Use authentic images for empirical claims. The existing Panda pair comes from the same selected video time; the ALOHA example uses a different held-correction protocol. A numerical failure panel provides actual failure evidence because the available corrected clips do not provide a verified failure photograph for that condition. Never manufacture a success or failure screenshot. State scene selection, protocol, frame/time, outcome provenance, and any crop. A video illustration is not a benchmark sample-size claim.

**Captions should be self-contained but layered:** a short takeaway; what panels/marks represent; essential units, sample size and uncertainty; the qualification needed to prevent the obvious misreading. Do not repeat every visible axis label in prose. Refer to panels in the same order in the text and caption. Keep terminology, method colors, and healthy/fault labels consistent across figures.

The examples show why two adjacent displays can be valuable when they answer different questions: [Safety Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/file/88be023075a5a3ff3dc3b5d26623fa22-Paper-Conference.pdf) separates a diagnostic from safety evaluation; [World-In-World](https://proceedings.iclr.cc/paper_files/paper/2026/file/5b4263be85820683d78675cc18d2efc7-Paper-Conference.pdf) relates intermediate measurements to closed-loop outcomes. This is the role of our task-success and tracking displays.

## 8. Plot precisely and make tables easy to audit

These are practical plotting choices, not statistical conclusions from the corpus:

- Use vector PDF/SVG for axes, lines, markers, equations and heatmap cells; use raster only for actual images. Embed fonts and inspect the final compiled PDF, not just a PNG preview.
- Design at final physical width. For this template, figures are approximately 5.5 inches wide. Prefer 8–10 point explanatory labels where feasible; simplify the panel before shrinking its text. A label readable only when zoomed is not a compact figure.
- Use a consistent, color-accessible palette and a second cue such as marker shape, line style, label or hatching. Put a diverging scale at zero for signed changes and retain the same limits across comparable panels.
- Prefer a condition-by-joint matrix over dozens of small bars when the question is heterogeneity. Put a numeric annotation in each cell if there are few enough cells; define whether the value is a count, percentage or percentage-point difference.
- Use direct paired differences with intervals when the scientific question is a paired comparison. Independent error bars for two means do not display the uncertainty of their paired difference.
- Label SD, SEM, confidence interval, bootstrap unit, and number of independent runs accurately. For clustered seeds reused across conditions, resample the seed cluster when that matches the estimand.
- Use shared axes and equal geometric aspect where required by the claim. A geometric norm illustration becomes misleading if one coordinate is stretched. A line plot should show the relevant reference threshold and axis units.
- Avoid radar charts for precise small differences, 3D bars, decorative gradients, and truncated bar baselines that exaggerate improvement. A log axis is useful for orders of magnitude but must be labeled.
- Show individual points or distribution summaries when variability is the question. If a curve is smoothed, state the window and show uncertainty from independent runs rather than treating time steps as independent replicates.
- Preserve an accessible data table or source JSON and deterministic plotting script. Check hashes, all conditions, units, arithmetic and rendered colors. Our prior PDF heatmap issue is why the current generator validates the actual PDF cells and color scale after rendering.

**Tables:** use meaningful row groups, aligned numeric columns, units in headers, counts/denominators, consistent precision and a small number of horizontal rules. State the relevant protocol when grouping reported and reproduced baselines. Use explicit missing-value markers. A broad table should show per-task evidence when averaging would hide failure; a second compact table can foreground the primary comparison and an important counterexample. Do not use a table as a paragraph broken into tiny cells.

For the current paper, Table 1 supplies exact counts and comparison definitions for selected physical-fault contrasts while Figure 3 shows the whole map. Table 2 gives raw healthy/faulted counts and the defined balanced score while Figure 3 supplies the primary difference interval. These functions complement one another. Their captions must not imply that an improvement over a harmful comparator establishes universal repair.

## 9. Allocate nine main pages by argumentative importance

The [current ICLR 2027 author instructions](https://iclr.cc/Conferences/2027/AuthorGuidelines) allow at most nine initial main-text pages, excluding references and appendices. The public proceedings PDFs studied here include later versions; their main-text lengths are not a waiver of the initial-submission limit. The required AI-use statement and recommended reproducibility statement are separately excluded under those instructions. Reviewers are not required to read the appendix.

A **starting editorial budget**, including each section's figures/tables, is 1.8 pages for title/abstract/introduction/overview; 0.8 for related work; 1.15 for method; 1.65 for conditional analysis; 3.2 for experiments; and 0.4 for limitations/conclusion. This totals nine pages. It is a planning device, not a measured average from award papers. The final draft is checked after float placement rather than relying on this estimate.

| Keep in the main paper | Expand in the appendix |
|---|---|
| Problem, closest prior work, exact contribution and data assumptions | Wider literature taxonomy and secondary connections |
| Executed update, fault/interface definitions, crucial timing and normalization | Full covariance recursion, bias/FIR derivations, pseudocode and implementation safeguards |
| Main theorem assumptions, statement, interpretation, one proof idea/example | Full proofs, extra lemmas and counterexamples |
| Robot/backbone/task, principal fault settings, correction support, split/selection protocol | Complete hyperparameter bank, selected configurations, rejected candidates, per-run manifests |
| Primary comparison, uncertainty, major negative result, scope-limiting control | Full per-condition tables, all historical profiles/backbones, supplementary diagnostics |
| Authentic overview examples and central quantitative figures | More frames, video provenance, additional qualitative examples |
| Calibration overlap, unresolved superiority, important confounding and unverified hardware scope | Detailed rerun plan and artifact-level audit records |

Do not move a serious limitation to the appendix simply because it weakens the headline. Conversely, source hashes, file paths, complete seed lists and implementation debugging history usually interrupt the main argument and can be indexed in the appendix or supplementary artifacts.

**Full is not crowded.** Use the page budget for a needed explanation, comparison or analysis. Trim duplicate summaries, repeated caveats at unrelated locations, generic background and redundant figure/table descriptions. Keep each important caveat where it changes interpretation. Do not alter official margins, line spacing or fonts, add negative spacing to hide overflow, or introduce filler to reach page nine. The final validation counts deferred main-text floats before the explicit backmatter boundary.

## 10. How many references, and where?

There is no defensible universal citation quota in these examples. Four manually checked bibliographies contain **34** works (*Vision Transformers Need Registers*), **37** (*Diffusion Geometry*), **44** (*Transformers are Inherently Succinct*) and **46** (*Differentiable MPC on the GPU*). The 2025 close readings find between **1 and 15 distinct cited works in selected first introduction paragraphs**, with the exact windows recorded in `reading_notes_2025.json`. These small, deliberately selected observations are illustrations of variation, not an estimated distribution for all ICLR papers.

The current manuscript cites **28 distinct works** from its verified bibliography. The writing revision preserves those citations. Increasing that number is justified only if a relevant foundation, nearest competitor, benchmark, implementation, or recent result is missing. A target of 50 or 80 would encourage padding rather than a better field map. Check the coverage axes in Section 3 before deciding whether more references are necessary.

| Location | What to cite | What the citation should support |
|---|---|---|
| Introduction | Foundational systems, precise prior limitation, nearest related attempt | Why the problem exists and what is already known |
| Related work | Representative families and closest competing constructions | The relationship asserted in that sentence, not merely the topic |
| Method | Inherited observer/update, representation, optimization routine | Attribution of the reused component and its actual assumptions |
| Theory | Borrowed theorem, standard result when helpful, established mathematical tool | The dependency in the argument; original derivations are explained and proved |
| Experiments | Dataset/robot platform, policy checkpoint family, baseline source, adopted protocol | Which system or procedure produced the comparison |
| Figure/table caption | Adapted material, an external dataset or previously reported number | Provenance where the display could otherwise look newly generated |
| Appendix | Additional related work, implementation references, detailed derivations | Expanded support without hiding the closest antecedent |

Use grouped citations when a sentence truly concerns a family; use an individual citation beside a specific comparison. Place citations close to the supported claim. Verify title, complete authors, publication venue/year, and DOI or proceedings record against primary metadata. Keep preprints labeled as preprints and distinguish conference event year from publication year. Do not invent BibTeX or claim Google Scholar provenance when the record actually comes from an official proceedings site or publisher.

## 11. What was changed in this manuscript

The revision applies the study's recommendations directly:

- **Abstract and introduction:** lead with the estimation-to-compensation gap; identify the two maps involved in VLA deployment; use the historical joint-5 harm as motivation with its different protocol stated; attach each contribution to a concrete theoretical or empirical deliverable.
- **Related work:** replace a sequence of model descriptions with four organizing questions about adaptation location, learned information, closest physical-fault evidence and control objectives. Preserve MAGIC-VFM, HMAC, Neural-Fly and the nearest fault-related work with their verified citations.
- **Method:** state offline fits versus online state evolution before the equations; annotate the composite update's prediction and tracking components; retain action/observation timing and the Kalman reduction. Qualify the target-normalized fixed point by open gating and inactive clipping.
- **Experiments:** replace generic subsection headings with the actual findings; state the three questions being tested; keep development, confirmation and diagnostics distinct. Describe tracking as an incomplete success proxy, rather than claiming its failure mechanism is settled.
- **Figures and tables:** retain the four complementary, reproducible main figures and exact-count tables. Shorten the overview caption while preserving the policy/adapter distinction, fault locations and video provenance. Keep the harmful conditions and primary uncertainty visible. Regenerate the three-robot appendix trace figure from pinned historical logs, fix its clipped GR1 title, and distinguish the held final-window average from a convergence-time claim.
- **Main/appendix split:** move implementation-debugging detail out of the main tracking paragraph where the same evidence is already in the tuning appendix; retain the confounding and missing matched-parameter ablation in the main text. Remove redundant concluding qualifications while preserving the important scope limits.
- **Layout and integrity:** rebuild the actual PDF; check nine main pages and unchanged official style; verify citations, float boundaries and rendering. The final master report records the resulting receipt.

No benchmark outcomes, significance tests, tuning results, proof claims or reference metadata were changed to strengthen the narrative. This task improves exposition; it does not establish composite superiority or repair every physical fault.

The accompanying merge of newer main-branch work retains GR1 scene-generator reseeding and the revised demonstration video. The appendix states the recorded innovation update and distinguishes future comparisons using the reset change from the historical unpaired cohorts. The manuscript retains Fisher tests for those historical cohorts; a reset fix and selected video do not create new paired benchmark results.

## 12. What writing cannot finish

The strongest remaining revision is experimental: an untouched matched-parameter tracking/damping ablation, truly separated calibration where historical claims require it, and renewed friction/lock evaluations if those efficacy claims are retained. Broad composite transfer and hardware claims need their own evidence. The current main text says so.

A professional submission should let a skeptical reader identify the contribution, reconstruct the implemented method, understand the evidence unit and comparison, locate the important limitation, and find the complete supporting artifact. Meeting these standards improves the paper's clarity and reviewability. It is not a certification that the results meet any conference's acceptance threshold.
