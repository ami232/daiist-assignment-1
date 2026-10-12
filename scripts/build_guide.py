"""Build the separate AI-generated study guide from the actual saved evidence.
Optional: uv run --with reportlab python scripts/build_guide.py
"""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/Project_Explanation_and_Defense_Guide.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
meta = json.loads((ROOT / 'artifacts/metadata.json').read_text())
with (ROOT / 'artifacts/metrics.csv').open() as f:
    metrics = list(csv.DictReader(f))
with (ROOT / 'artifacts/feature_experiments.csv').open() as f:
    experiments = list(csv.DictReader(f))
def result(split, name):
    return next(r for r in metrics if r['split'] == split and r['model'] == name)
r = result('test','sklearn')
v = result('validation','sklearn')
def n(k): return int(float(r[k]))
def money(x): return f'EUR {float(x):,.0f}'
navy, teal, pale = colors.HexColor('#152B3B'), colors.HexColor('#007F73'), colors.HexColor('#EDF5F3')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX', fontName='Helvetica-Bold', fontSize=31, leading=35, textColor=navy, spaceAfter=18))
styles.add(ParagraphStyle(name='KickerX', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=teal, spaceAfter=10))
styles.add(ParagraphStyle(name='HeadX', fontName='Helvetica-Bold', fontSize=21, leading=26, textColor=navy, spaceAfter=16))
styles.add(ParagraphStyle(name='SubX', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=teal, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name='BodyX', fontName='Helvetica', fontSize=10.4, leading=15.2, spaceAfter=9, textColor=navy))
styles.add(ParagraphStyle(name='SmallX', fontName='Helvetica', fontSize=8.5, leading=11.5, spaceAfter=6, textColor=navy))
styles.add(ParagraphStyle(name='CodeX', fontName='Courier', fontSize=9, leading=13, backColor=pale, borderPadding=9, spaceBefore=7, spaceAfter=13))
styles.add(ParagraphStyle(name='CellX', fontName='Helvetica', fontSize=8.6, leading=11.5, textColor=navy))
styles.add(ParagraphStyle(name='CellHeadX', fontName='Helvetica-Bold', fontSize=8.6, leading=11.5, textColor=colors.white))
story=[]
REVISIONS = json.loads((ROOT / 'docs/walter_guide_revisions.json').read_text())
def p(text, style='BodyX'):
    for prefix, revised in REVISIONS.items():
        if text.startswith(prefix):
            text = revised
            break
    story.append(Paragraph(text, styles[style]))
def h(text): p(text,'SubX')
def code(text): p(escape(text).replace('\n','<br/>'),'CodeX')
def page(num,title):
    if story: story.append(PageBreak())
    p(f'PROJECT EXPLANATION / {num:02d}', 'KickerX'); p(title,'HeadX')
def table(headers, rows, widths):
    data=[[Paragraph(escape(str(x)),styles['CellHeadX']) for x in headers]]
    data += [[Paragraph(escape(str(x)),styles['CellX']) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),('LEFTPADDING',(0,0),(-1,-1),8),
                          ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                          ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#CCD9DA'))]))
    story.append(t);story.append(Spacer(1,10))

def footer(canvas,doc):
    canvas.setStrokeColor(teal);canvas.line(43,43,552,43)
    canvas.setFont('Helvetica',8);canvas.setFillColor(navy)
    canvas.drawString(43,29,'Javier Sanchez Esquivel | Assignment 1 | AI-generated study guide')
    canvas.drawRightString(552,29,f'{doc.page}')

page(1,'Bank marketing')
p('From a business decision<br/>to three equivalent models', 'TitleX')
p('Explanation and professor-discussion guide', 'SubX')
p('Javier Sánchez Esquivel · Student 19731 · BBADBA 5B<br/>jsanchez.ieu2022@student.ie.edu · 12 October 2026')
p('This guide explains the implemented project, its measured results, the reasoning behind the student-selected decisions, and the concepts needed to discuss it without looking at the code. It is an explicitly AI-generated learning aid, separate from the student-authored interpretation in REPORT.md.')
table(['What was built','Verified outcome'],[
 ['Data and features','41,188 historical bank-marketing records; chronological evaluation; 46 encoded pre-call features.'],
 ['Three implementations','scikit-learn, manual PyTorch and nn.Module + SGD converge to nearly identical probabilities.'],
 ['Interactive dashboard','Saved-model inference, model comparison, prediction-versus-actual views, distributions, confusion matrix and threshold costs.'],
 ['Business result',f'Test policy cost {money(r["cost_eur"])}; calling everyone costs EUR 16,476. Correct code does not imply a useful policy.'],
 ['Submission boundary','Fork and upstream PR are prepared separately from Blackboard. See docs/DELIVERY.md for current links and status.']],[143,366])
h('How to study this guide')
p('First understand the business arithmetic on pages 2 and 8. Then explain the feature meanings on page 4 and the shared model on pages 5-6. Finally rehearse the demonstration and questions on pages 10-12. You should be able to explain the tradeoffs in your own words, not memorize a script.')

page(2,'Start with the decision')
p('The student chose a bank calling team that must prioritize customers before a planned call. The binary target is whether a customer subscribed to a term deposit in the historical campaign. The output is a probability, followed by a separate call/no-call decision.')
h('What can be known at that moment?')
p('The project assumes the customer profile and previous-campaign history are in the CRM before the decision. It excludes call duration because duration is only available after a call. It also excludes the completed campaign count and final-call channel, month and weekday, since the file does not establish that these describe the planned next call.')
p('Economic indicators are excluded conservatively: the dataset gives observed values but not release-time or revision-vintage information. Excluding them sacrifices potentially useful signal. It is an explicit availability choice, not a claim that macroeconomic conditions are irrelevant.')
h('The student-selected cost assumptions')
code('Cost = EUR 2 * (TP + FP) + EUR 50 * FN')
table(['Term','Business meaning'],[
 ['TP','We call a historical subscriber. This call still costs EUR 2.'],['FP','We call a historical non-subscriber. The call costs EUR 2.'],
 ['FN','We skip a historical subscriber. Assign a hypothetical EUR 50 missed-opportunity cost.'],['TN','We skip a historical non-subscriber. No call cost in this simplified model.']],[62,447])
p('For example, 20 calls and 3 misses cost 20 × 2 + 3 × 50 = EUR 190. Accuracy is not the policy objective: a false negative is much more expensive here than an unnecessary call.')
h('A distinction to say clearly')
p('This is response prediction, not causal uplift modeling. Everyone in the historical campaign was exposed to marketing in some form. We do not observe what would have happened without calling, so we cannot claim the policy creates deposits or measures actual profit.')

page(3,'Understand the data and split')
p('The project uses the original UCI <b>bank-additional-full.csv</b>: 41,188 records, 20 input columns and the target y. A row is a client/campaign record with customer information and summarized contacts; it is not a transaction or necessarily a unique person across the dataset.')
p('The supplied dictionary states that rows are ordered by date from May 2008 to November 2010. Exact dates and customer IDs are absent. We retain row order rather than inventing dates or sorting by month, which would mix years.')
table(['Block','Rows / zero-based IDs','Subscriber rate'],[
 ['Training','24,712 / 0-24,711','4.807%'],['Validation','8,238 / 24,712-32,949','11.071%'],['Test','8,238 / 32,950-41,187','30.833%']],[107,266,136])
h('What each block is allowed to do')
p('<b>Training:</b> learn imputations, means/standard deviations, one-hot vocabulary and model parameters. <b>Validation:</b> compare a small set of feature/regularization choices and choose the threshold. <b>Test:</b> evaluate the already frozen choices. Moving the test threshold to get a nicer result would invalidate that final comparison.')
h('Why chronology matters')
p('The intended use is later campaigns. Randomly mixing early and late records could give training access to future conditions. Chronology is therefore more relevant, even though its results are less flattering. Row boundaries may split a day or campaign because exact timestamps are unavailable; repeated-client separation also cannot be guaranteed.')
h('Data-quality facts')
p('There are no blank cells, but categorical "unknown" values still represent missing information. Default has 8,597 unknowns; education 1,731; housing and loan 990 each. The file has 12 exact duplicate rows. They are retained: without identifiers, identical recorded attributes are insufficient evidence that a record is erroneous.')
p('The raw data remains unchanged and bundled locally. A SHA-256 in metadata.json records its exact identity. The full file is only 5.8 MB, despite exceeding the assignment\'s approximate 20,000-row guideline.')

page(4,'Explain every engineered feature')
p('The original numeric age and categorical customer attributes remain. Categorical values use one-hot encoding. Poutcome retains previous-campaign outcome. The selected feature recipe adds the following:')
table(['Feature','Construction','Reason and limitation'],[
 ['recency_recorded','1 if pdays is not 999; otherwise 0','Separates a recorded interval from the 999 sentinel; it does not establish absence of all past contacts.'],
 ['log_recency_days','log(1 + pdays) when observed; otherwise 0','Compresses elapsed days; the indicator distinguishes the zero placeholder from a real 0-day interval.'],
 ['log_previous_contacts','log(1 + previous)','Compresses a skewed count. With 0 or 1 previous contact in training, behavior for larger later counts is extrapolation.'],
 ['loan_combination','housing + / + loan','Represents combinations, including unknown. Allows a joint association beyond separate main effects.'],
 ['Unknown-field count','Count of unknown among six customer categories','Allows incomplete records to carry a common signal; missingness can change over time.']],[115,158,236])
h('An example you can explain')
p('If pdays=999 and previous=0, the recency indicator is 0, log recency is 0, and log previous contacts is 0. If pdays=3 and previous=2, they become 1, log(4)=1.386 and log(3)=1.099. These are feature values before numeric standardization.')
p('Data-quality caveat: 4,110 records have pdays=999 despite previous&gt;0. The indicator therefore means recency recorded, not never contacted. The count and prior outcome are preserved instead of silently rewriting the history.')
h('Preprocessing')
p('Numeric values are median-imputed defensively, then standardized using training statistics: z=(x-training mean)/training standard deviation. No numeric values are missing in this file, so imputation does not change the observed dataset. Unknown categories are retained rather than silently converted to the most frequent known category. An unseen category is safely ignored by the fitted encoder, but that does not mean the model understands it.')
p('The selected recipe has 46 encoded columns. Neither the target nor excluded fields can enter the feature matrix: the implementation whitelists permitted columns. Automated tests perturb forbidden fields and verify that features do not change.')

page(5,'The model and loss')
p('All three implementations are <b>logistic regression</b>. There are no hidden layers, trees or neural-network feature extractors. PyTorch is a framework; using it does not automatically make the model deep learning.')
code('logit z = x @ w + b\np = sigmoid(z) = 1 / (1 + exp(-z))\ncall if p >= threshold')
p('The weighted sum z can be any real number. The sigmoid maps it to 0-1. The intercept b is the score when the encoded numeric inputs are at their training means and other encoded inputs are zero; with full one-hot encoding, do not interpret it as a literal typical customer.')
h('The shared training objective')
code('J(w,b) = mean[softplus(z) - y*z] + (lambda/2) * sum(w**2)\nsoftplus(z) = log(1 + exp(z)), computed stably')
p('The first term is binary cross-entropy, also called log loss. It penalizes confident wrong probabilities strongly. The second term is L2 regularization: it discourages large weights and helps with redundant or correlated encoded features. The intercept is not penalized. There are no class weights; changing class weights would change the objective and complicate probability interpretation.')
h('What the gradients mean')
code('error = sigmoid(X @ w + b) - y\ngrad_w = X.T @ error / n + lambda * w\ngrad_b = mean(error)')
p('A gradient tells us how the objective changes with each parameter. Gradient descent subtracts learning_rate × gradient. Autograd computes these derivatives from the computation graph. The tests independently compare autograd with the analytic formulas and finite differences.')
h('Scaling and feature engineering are different')
p('Scaling changes numeric units for optimization and regularization. Engineering changes what relationships the model can express. A linear score on transformed inputs can represent log-count effects and category interactions while remaining logistic regression in its parameters.')

page(6,'Three implementations, one objective')
table(['Implementation','What happens','What to point to'],[
 ['scikit-learn','LogisticRegression uses L-BFGS to minimize the regularized objective.','train.py: C conversion, fit, predict_proba'],
 ['Manual PyTorch','Create raw trainable tensors. Compute stable loss, backward, and update tensors inside no_grad. Clear gradients after each step.','train.py: objective and train_torch(standard=False)'],
 ['Standard PyTorch','nn.Linear produces one logit. BCEWithLogitsLoss and explicit L2 penalty define the loss; optimizer.zero_grad, backward, optimizer.step perform updates.','modeling.py; train_torch(standard=True)']],[113,244,152])
h('Regularization equivalence is essential')
code(f'PyTorch: mean BCE + lambda/2 * ||w||^2\nscikit-learn: mean BCE + 1/(2*C*n) * ||w||^2\nTherefore C = 1/(n*lambda) = {meta["sklearn_C"]:.10f}')
p('Using the same numeric C and lambda would not produce the same objective. The equivalence test evaluates the shared PyTorch gradient at scikit-learn\'s solution and requires it to be close to zero.')
h('Learning rate and stopping')
p(f'Both PyTorch models start with zero weights and zero intercept, use full batches and float64 on CPU, and use the same learning rate {meta["learning_rate"]:.6f}. It is 1/L, with L computed from a conservative curvature bound on the training matrix. This is a stable training-based step choice, not a search over test scores. Both stop after 6,433 updates when the largest absolute training gradient falls below 1e-7.')
p('In more detail, augment X with a column of ones for the intercept. For logistic loss, curvature is bounded by 0.25 times the largest eigenvalue of X_aug.T @ X_aug / n. Adding lambda gives a conservative bound for the penalized objective.')
h('Why tiny differences remain')
p(f'Maximum probability disagreement against scikit-learn is {meta["agreement"]["manual"]["max_probability_difference_all_rows"]:.8f}; test call decisions disagree on zero records. Different numerical solvers and finite stopping tolerance account for small differences. Ranking metrics can also shift minutely when nearly tied scores swap order.')

page(7,'Experiments and measured results')
p('There were three feature recipes and three L2 strengths (0.001, 0.01 and 0.1), evaluated on the same validation block. This is a small, inspectable comparison rather than a large hyperparameter search. The held-out test was not used to select these choices.')
best_by={}
for item in experiments:
    if item['variant'] not in best_by: best_by[item['variant']]=item
table(['Recipe','Best lambda','Validation log loss'],[[name,best_by[name]['lambda'],f'{float(best_by[name]["validation_log_loss"]):.6f}'] for name in ['raw','history','history_context']],[245,102,162])
p('The student selected history_context despite the raw control\'s small score advantage, because the engineered representation gives the sentinel its intended meaning. The validation loss difference is about 0.00158. This is not evidence that engineered features improved prediction, nor a statistical significance claim. Adding loan/unknown context slightly helps relative to history alone at lambda=0.01.')
h('Final test: probability quality')
table(['Method','ROC-AUC','Avg precision','Log loss','Brier'],[[name,f'{float(result("test",name)["roc_auc"]):.6f}',f'{float(result("test",name)["average_precision"]):.6f}',f'{float(result("test",name)["log_loss"]):.6f}',f'{float(result("test",name)["brier"]):.6f}'] for name in ['sklearn','manual','standard','baseline']],[117,94,104,97,97])
p('<b>ROC-AUC:</b> how often a subscriber ranks above a non-subscriber. <b>Average precision:</b> a precision-recall summary, useful for imbalanced classes; here the constant baseline is the test subscriber rate, 0.308327. <b>Log loss:</b> probability error with strong penalties for confident mistakes. <b>Brier:</b> mean squared probability error.')
p('The baseline always predicts the training subscriber prevalence, 0.048074. At the low chosen threshold it calls everyone. It is fitted on training only; using test prevalence as its prediction would leak test labels into the baseline.')
h('Training speed is secondary evidence')
p(f'This saved run recorded approximately {meta["training_seconds"]["sklearn"]:.3f} seconds for the selected scikit-learn fit, {meta["training_seconds"]["manual"]:.2f} seconds for manual PyTorch, and {meta["training_seconds"]["standard"]:.2f} seconds for standard PyTorch. These are individual fit timings on this Mac, excluding preprocessing and the small model-selection experiment. They are not a controlled benchmark and will vary.')

page(8,'Thresholds and the business result')
p(f'The final frozen threshold is <b>{meta["threshold"]:.8f}</b>. Validation selects the lowest assumed cost among attainable call policies. Ties favor fewer calls. The implementation groups equal scores and places the boundary between adjacent score levels to avoid floating-point equality problems after saving and loading.')
p('If probabilities were perfectly calibrated and the simplified cost assumptions were correct, calling costs EUR 2 and skipping costs 50 × p in expectation. Thus calling is preferred when p exceeds 2/50=0.04. The empirical validation threshold is nearby but not identical; it reflects finite validation observations and imperfect probability calibration.')
table(['Policy / test block','Calls','Misses','Assumed cost'],[
 ['Frozen model policy',f'{n("calls"):,}',f'{n("fn"):,}',money(r['cost_eur'])],['Call everyone','8,238','0','EUR 16,476'],['Call nobody','0','2,540','EUR 127,000']],[200,80,80,149])
code(f'Saved calls = 8,238 - {n("calls"):,} = {8238-n("calls")}\nSaved call cost = {8238-n("calls")} * EUR 2 = EUR {(8238-n("calls"))*2}\nMissed opportunity = {n("fn")} * EUR 50 = EUR {n("fn")*50:,}\nExtra cost vs call-all = EUR {n("fn")*50-(8238-n("calls"))*2:,}')
p(f'The policy recalls {float(r["recall"]):.2%} of historical subscribers but calls {n("calls")/8238:.2%} of all records. High recall alone does not demonstrate efficient prioritization. Under these costs, missing even a few subscribers can outweigh saving many calls.')
h('Validation versus test')
p(f'On validation, the selected policy costs {money(v["cost_eur"])} versus EUR 16,476 for calling everyone: only EUR {16476-float(v["cost_eur"]):.0f} lower. That small advantage does not generalize to test. The dashboard must expose this outcome, not optimize the test threshold after observing it.')
p('Changing the dashboard costs or threshold is an explanatory what-if exercise. It does not change the saved models or the frozen report results. A new deployed policy would need a separate validation process and later unseen evaluation.')

page(9,'What the evidence cannot establish')
h('Substantial change across chronological blocks')
p('Subscriber prevalence rises from 4.807% to 11.071% to 30.833%. Previous-contact histories also change: training contains only previous=0 or 1, while later blocks contain larger counts. The model therefore extrapolates beyond its training experience. These are observed distribution differences; this project does not establish which changes caused the loss of policy value.')
h('Probability calibration')
p('An AUC above 0.5 shows some ranking signal, but probabilities can still be poorly calibrated. Inspect the predicted-versus-observed plot: points off the diagonal mean predicted probabilities differ from observed subscriber fractions in those bins. Sparse bins have high sampling uncertainty. Do not interpret a bin based on very few examples as a precise estimate.')
h('Scope and missing information')
p('The sample is one bank\'s historical marketing population from 2008-2010, not current bank customers in general. Exact contact dates, stable customer IDs, cost/revenue measurements and randomized no-call outcomes are absent. A row-order split cannot enforce customer disjointness or a full time gap between campaigns.')
h('Features are hypotheses, not guarantees')
p('The tested engineering improves meaning and explicit handling of missing/sentinel values, but did not beat the raw control on validation log loss. Some training features are rare, including recorded previous-contact intervals. Large later values or different missingness patterns can change model behavior. Coefficients represent conditional associations; correlated inputs and L2 shrinkage prevent causal interpretation.')
h('How a future investigation would remain honest')
p('A follow-up could assess successive later campaign windows, check calibration using a recent validation block, verify the real value and cost assumptions, and obtain customer IDs and exact timestamps. Those changes would need a new untouched evaluation period. The existing test results should remain documented instead of being silently replaced by a tuned result.')
h('Your conclusion belongs to you')
reflection=ROOT/'docs/STUDENT_REFLECTION.txt'
if reflection.exists():
    p('The student\'s supplied conclusion, preserved separately from this guide\'s teaching explanations:')
    p('I would not deploy this policy as is. The most useful next step is a temporal backtest with a calibration check on successive later campaign blocks. Real cost and revenue estimates should replace the hypothetical costs before the policy is reconsidered. The full student reflection and verified arithmetic are preserved in REPORT.md.')
else:
    p('Before submitting, state your own recommendation in REPORT.md: whether you would use this exact policy under the chosen assumptions and which next investigation you consider most useful. The code and numbers cannot make that judgment for you.')

page(10,'Run and demonstrate the project')
code('uv sync --frozen\nuv run python main.py train\nuv run python -m pytest -q\nuv run python main.py app')
p('Run these from the project folder, then open http://127.0.0.1:7860. The first dependency installation needs Internet access. Training itself reads the included CSV and makes no network requests. Stop the app with Ctrl+C. The required main.py entrypoint remains unchanged.')
table(['Demonstration step','What to show and explain'],[
 ['1. Overview','Read the final test result and the cost comparison. Explain why saved call costs do not cover the missed-subscriber penalty.'],
 ['2. Call policy','Start on validation. Use Call everyone (0%), Call nobody (100%), and Restore saved policy. Switch to test to check later campaigns.'],
 ['3. Models','Show that metrics agree across three models. Explain calibration and the actual 0/1 outcomes versus predicted probabilities.'],
 ['4. Data','Show subscriber prevalence by split and a prior-history feature. Explain chronological change and why 999 is a sentinel.'],
 ['5. Records','Select a record and predict. The app transforms its inputs and invokes loaded model objects; it does not simply fit a new model or train at startup.']],[121,388])
h('Architecture in one sentence')
code('local CSV -> train-only preprocessing -> 3 fitted models\n            -> saved artifacts -> read-only Gradio inference')
p('Training saves the preprocessing object, scikit-learn model, raw PyTorch tensors, standard PyTorch state_dict and the naive baseline. It also exports predictions, metrics, coefficients, split membership, experiment results and training traces. The dashboard loads these files; app.py never calls a fitting routine.')
p('The model pickles are locally created trusted artifacts. Do not replace them with an untrusted downloaded pickle. Rebuilding from the bundled data is the reproducible path.')

page(11,'Verification and grading coverage')
table(['Grading component','Evidence to point to'],[
 ['Dataset & business framing: 20%','Pre-call decision, EUR 2/EUR 50 assumptions, chronology, information-availability exclusions; REPORT.md and DECISIONS.md.'],
 ['Features & preprocessing: 30%','Explicit sentinel semantics, log history, loan interaction, unknown count, training-only transformations; all nine validation configurations retained.'],
 ['Three-method comparison: 30%','Same objective and data, correct C/lambda mapping, shared PyTorch step/stopping, numerical agreement, naive baseline and candid cost comparison.'],
 ['Gradio dashboard: 20%','Working interactive views, actual saved models, confusion matrix and business cost respond to threshold changes.'],
 ['Written comprehension multiplier','Explain both the implementation and its limitations without the code. This guide supports preparation; no grade is guaranteed.']],[172,337])
p('These are the weights stated in the repository when inspected; the brief describes them as a proposal subject to confirmation. A hard requirement is that both main.py commands run without manual intervention. REPORT.md supplies evidence for the weighted components rather than receiving a separate row.')
h('What the automated checks protect against')
p('The checks verify: forbidden fields cannot influence features; sentinel and unknown handling; chronological partition and training-only statistics; analytic/finite-difference gradient agreement; scikit-learn objective equivalence; persisted predictions and inter-method agreement; threshold endpoints and tied scores; validation-only threshold selection; app inference with fit methods disabled; and exact reproduction of exported costs in the dashboard.')
h('What local verification does not claim')
p('The project has been trained and tested on this Mac. Browser/API checks verify the local Gradio app. The original GitHub Actions workflow is included. See docs/DELIVERY.md for the latest remote verification status. This is not proof of performance at another bank, causal marketing lift, or a guaranteed grade.')
h('Backup and submission boundary')
code('uv run python scripts/package_project.py')
p('This creates a local ZIP beside the project folder, excluding the environment, Git internals and caches. It never submits. The authorized GitHub steps are pushing the fork and opening the upstream PR. Providing the PR link and ZIP on Blackboard remains for the student. See docs/DELIVERY.md for links and status.')

page(12,'Practice without your code')
table(['Professor question','A complete answer should include'],[
 ['Why logistic regression?','Binary target; a weighted sum mapped through sigmoid; interpretable baseline; assignment asks for the same simple model in three frameworks.'],
 ['Why is duration leakage?','It is unknown before the call. Good test scores using unavailable inputs would not validate the intended decision.'],
 ['Why not a random split?','Later-campaign use and changing populations. Preserve source chronology; acknowledge missing exact timestamps and IDs.'],
 ['What does backward do?','Computes gradients. It does not update weights. Manual no_grad updates or optimizer.step apply the update.'],
 ['Why zero gradients?','PyTorch accumulates gradients; failing to reset them changes the optimization procedure.'],
 ['Why BCEWithLogitsLoss?','Stable combination of sigmoid and BCE. Feed logits, not already-sigmoided probabilities.'],
 ['Why do all models agree?','Same X, y, regularized objective and converged optimization. Their agreement is a correctness check, not a policy-value guarantee.'],
 ['Why not threshold 0.5?','Decision costs are asymmetric. A subscriber miss costs much more than a call; validation selects the policy threshold.'],
 ['Why is test accuracy low?','The cost-optimal validation policy calls almost everyone. Accuracy values error types equally; chosen costs do not.'],
 ['Can you claim feature engineering helped?','It improved explicit semantics; measured validation log loss was slightly worse than the raw control. Preserve that distinction.'],
 ['Would you deploy?','Give your own recommendation, cite the EUR 1,406 test disadvantage and explain what further evidence you require.']],[171,338])
h('Three short exercises')
p('<b>A.</b> With 100 calls and 4 missed subscribers, what is policy cost? <b>B.</b> If lambda doubles at fixed n, what happens to scikit-learn C? <b>C.</b> If two frameworks agree exactly, can both be wrong for the business?')
p('<b>Answers:</b> A: EUR 400. B: C halves. C: Yes; they may share poor features, distribution mismatch, invalid business assumptions or an unsuitable decision rule. Agreement alone does not validate business value.')

page(13,'Sources and your final review')
h('Primary sources')
p('1. Assignment brief and rubric: <link href="https://github.com/ami232/daiist-assignment-1" color="#007F73">github.com/ami232/daiist-assignment-1</link>. Local copy: docs/ASSIGNMENT.md.', 'SmallX')
p('2. UCI Bank Marketing: <link href="https://archive.ics.uci.edu/dataset/222/bank+marketing" color="#007F73">archive.ics.uci.edu/dataset/222/bank+marketing</link>. Dataset DOI: 10.24432/C5K306. Creators: Moro, Rita and Cortez. License: CC BY 4.0. The local bank-additional-names.txt defines this version\'s fields.', 'SmallX')
p('3. Moro, Cortez and Rita (2014), A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems 62, 22-31. DOI: 10.1016/j.dss.2014.03.001.', 'SmallX')
p('4. scikit-learn logistic-regression objective: <link href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression" color="#007F73">scikit-learn.org/stable/modules/linear_model.html#logistic-regression</link>. PyTorch loss implementation: BCEWithLogitsLoss and functional.softplus in the installed torch package.', 'SmallX')
h('Evidence generated by this project')
p('All numerical results in this guide come from artifacts/metadata.json, metrics.csv, feature_experiments.csv and the committed source data. They are not borrowed benchmark results. The exact source hash is:', 'SmallX')
code(meta['source_sha256'][:32]+'\n'+meta['source_sha256'][32:])
table(['Package','Saved run version'],list(meta['versions'].items()),[255,254])
h('Before you explain the project')
p('Be able to reconstruct the EUR 1,406 comparison, explain why a cost-driven threshold is low, derive C=1/(n*lambda), and identify what information was excluded before a call. Open the dashboard and practice the five-step demonstration. Read your own report conclusion and make sure it says what you actually believe.')
h('AI disclosure and ownership')
p('Codex wrote and debugged implementation code, tests and dashboard presentation, assisted with evidence formatting, and authored this separate study guide. The student chose the topic, decision point, cost assumptions, split strategy, feature ideas and the semantic-versus-score tradeoff. REPORT.md must disclose this assistance and preserve the student\'s own analysis. Understanding still has to be demonstrated by the student.')
p('Blackboard submission has not been performed. The GitHub fork and PR are authorized preparation steps; the current links and status are recorded in docs/DELIVERY.md. The ZIP is a local copy for the student to upload.')

doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),rightMargin=43,leftMargin=43,topMargin=42,bottomMargin=57,
                      title='Bank Marketing - Project Explanation and Defense Guide',author='Javier Sanchez Esquivel; AI-assisted study guide')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
