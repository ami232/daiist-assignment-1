"""Saved-model Gradio dashboard. This module contains no fitting or training."""
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import gradio as gr
from features import engineer
from modeling import LogisticModel, tensor_probabilities
from evaluation import policy_counts

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
MODELS = ['sklearn', 'manual', 'standard', 'baseline']
COLORS = ['#087F73', '#5471C4', '#986819', '#89939F']
MODEL_LABELS = {'sklearn': 'Scikit-learn', 'manual': 'PyTorch · manual', 'standard': 'PyTorch · standard', 'baseline': 'Naive baseline'}
SPLIT_LABELS = {'train': 'Training · earliest', 'validation': 'Validation · middle', 'test': 'Test · latest'}
FEATURE_LABELS = {'age': 'Age', 'job': 'Occupation', 'marital': 'Marital status', 'education': 'Education',
 'default': 'Credit default status', 'housing': 'Housing loan', 'loan': 'Personal loan', 'poutcome': 'Previous campaign outcome',
 'recency_recorded': 'Previous-contact recency recorded', 'log_recency_days': 'Days since contact · log scale',
 'log_previous_contacts': 'Previous contacts · log scale', 'loan_combination': 'Housing + personal loan',
 'unknown_customer_fields': 'Unknown customer fields', 'actual_subscription': 'Customers who subscribed'}
CSS = (ROOT / 'assets/dashboard.css').read_text()


def dashboard_theme():
    theme = gr.themes.Soft(primary_hue='teal', neutral_hue='slate', font=['Arial', 'sans-serif'])
    values = {'body_background_fill': '#f5f4ef', 'body_text_color': '#192e35', 'body_text_color_subdued': '#52616b',
              'block_background_fill': '#f5f4ef', 'block_border_color': '#e0e7ea', 'block_label_background_fill': '#f5f4ef',
              'block_label_text_color': '#354b58', 'block_title_text_color': '#192e35', 'input_background_fill': '#f3f6f7',
              'input_border_color': '#ccd8dd', 'button_primary_background_fill': '#17675d', 'button_primary_text_color': '#ffffff',
              'button_secondary_background_fill': '#eaf1f2', 'button_secondary_text_color': '#192e35',
              'border_color_primary': '#d9e3e7'}
    return theme.set(**{k+suffix:v for k,v in values.items() for suffix in ['', '_dark']})



def style(fig, title=None):
    fig.update_layout(template='plotly_white', title=dict(text=title, font=dict(size=17)),
                      font=dict(family='Arial', size=12, color='#192e35'),
                      margin=dict(l=65, r=30, t=55, b=75), height=430,
                      colorway=COLORS, legend_title_text='',
                      legend=dict(orientation='h', y=-0.22, x=0),
                      paper_bgcolor='#f5f4ef', plot_bgcolor='#f5f4ef')
    fig.update_xaxes(automargin=True, showline=False, zeroline=False, gridcolor='#e0e4dc')
    fig.update_yaxes(automargin=True, showline=False, zeroline=False, gridcolor='#e0e4dc')
    return fig


class Dashboard:
    def __init__(self, artifact_dir=ARTIFACTS):
        self.path = Path(artifact_dir)
        if not (self.path / 'metadata.json').exists():
            raise RuntimeError('Saved models are missing. Run: uv run python main.py train')
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        self.raw = pd.read_csv(ROOT / 'data/bank-additional-full.csv', sep=';')
        self.predictions = pd.read_csv(self.path / 'predictions.csv', float_precision='round_trip')
        self.preprocessor = joblib.load(self.path / 'preprocessor.joblib')
        self.sklearn = joblib.load(self.path / 'sklearn.joblib')
        self.baseline = joblib.load(self.path / 'baseline.joblib')
        self.manual = torch.load(self.path / 'manual.pt', map_location='cpu', weights_only=True)
        self.standard = LogisticModel(self.meta['n_features'])
        self.standard.load_state_dict(torch.load(self.path / 'standard.pt', map_location='cpu', weights_only=True))
        self.standard.eval()
        self.features = engineer(self.raw, self.meta['variant'])
        self.metrics = pd.read_csv(self.path / 'metrics.csv')
        self.coefficients = pd.read_csv(self.path / 'coefficients.csv')

    def predict_rows(self, raw):
        x = self.preprocessor.transform(engineer(raw, self.meta['variant'])).astype(np.float64)
        with torch.no_grad():
            standard = torch.sigmoid(self.standard(torch.tensor(x, dtype=torch.float64))).numpy()
        return {'sklearn': self.sklearn.predict_proba(x)[:, 1],
                'manual': tensor_probabilities(x, self.manual['weights'], self.manual['bias']),
                'standard': standard, 'baseline': self.baseline.predict_proba(x)[:, 1]}

    def policy(self, model, split, threshold, call_cost, missed_cost):
        block = self.predictions[self.predictions.split == split]
        y, p = block.actual.to_numpy(), block[model].to_numpy()
        c = policy_counts(y, p, threshold, call_cost, missed_cost)
        fig = go.Figure(go.Heatmap(z=[[c['tn'], c['fp']], [c['fn'], c['tp']]],
            x=['Skip customer', 'Call customer'], y=['Did not subscribe', 'Subscribed'],
            text=[[f"{c['tn']:,}<br>Correctly skipped · TN", f"{c['fp']:,}<br>Unnecessary calls · FP"],
                  [f"{c['fn']:,}<br>Missed subscribers · FN", f"{c['tp']:,}<br>Subscribers called · TP"]],
            texttemplate='%{text}', textfont=dict(size=14), colorscale=[[0,'#eff5f4'],[1,'#348d81']], showscale=False,
            hovertemplate='%{y}<br>%{x}<br>%{z:,} customers<extra></extra>'))
        fig.update_yaxes(autorange='reversed', title='What happened historically')
        fig.update_xaxes(title='What the model recommends')
        fig = style(fig, 'Who gets called, and who gets missed?')
        return self.policy_summary(model, split, threshold, call_cost, missed_cost), fig, self.cost_comparison(model, split, threshold, call_cost, missed_cost)

    def policy_summary(self, model, split, threshold, call_cost, missed_cost):
        block = self.predictions[self.predictions.split == split]
        c = policy_counts(block.actual, block[model], threshold, call_cost, missed_cost)
        everyone = len(block) * call_cost
        difference = c['cost_eur'] - everyone
        direction = 'more' if difference > 0 else 'less' if difference < 0 else 'the same'
        verdict = f"€{abs(difference):,.0f} {direction} than calling everyone" if difference else 'Same cost as calling everyone'
        color = 'bank-warning' if difference > 0 else 'bank-good'
        skipped = len(block) - c['calls']
        saved = skipped * call_cost
        missed = c['fn'] * missed_cost
        recall = c['tp'] / max(c['tp'] + c['fn'], 1)
        frozen = abs(threshold-self.meta['threshold']) < 1e-12 and call_cost == 2 and missed_cost == 50
        state = 'Saved policy' if frozen else 'What-if scenario'
        return f"""<section class="bank-summary">
        <div class="bank-kicker">{state} · {SPLIT_LABELS[split]} · {len(block):,} records</div>
        <div class="bank-result {color}">{verdict}</div>
        <p>Estimated policy cost: <b>€{c['cost_eur']:,.0f}</b>. Calling every customer would cost <b>€{everyone:,.0f}</b>.</p>
        <div class="bank-stats">
          <div class="bank-stat"><strong>{c['calls']:,} calls</strong><span>{c['calls']/len(block):.1%} of customers selected</span></div>
          <div class="bank-stat"><strong>{c['fn']:,}</strong><span>subscribers missed by skipping</span></div>
          <div class="bank-stat"><strong>{recall:.1%}</strong><span>of subscribers reached · recall</span></div>
        </div>
        <div class="bank-equation">Skip <b>{skipped:,}</b> customers → save <b>€{saved:,.0f}</b> on calls.<br>
        Subscribers missed: <b>{c['fn']:,}</b> → add <b>€{missed:,.0f}</b> in assumed opportunity cost.</div>
        <p class="bank-note">Call at a predicted probability of {threshold:.2%} or higher. Assumptions: €{call_cost:g} per call and €{missed_cost:g} per missed subscriber. Historical cost illustration, not measured profit or causal impact.</p>
        </section>"""

    def cost_comparison(self, model, split, threshold, call_cost, missed_cost):
        block = self.predictions[self.predictions.split == split]
        c = policy_counts(block.actual, block[model], threshold, call_cost, missed_cost)
        labels = ['Call everyone', 'Selected policy']
        call_values = [len(block)*call_cost, c['calls']*call_cost]
        miss_values = [0, c['fn']*missed_cost]
        fig = go.Figure()
        fig.add_bar(y=labels, x=call_values, orientation='h', name='Cost of making calls', marker_color='#17675d',
                    hovertemplate='%{y}<br>Call cost: €%{x:,.0f}<extra></extra>')
        fig.add_bar(y=labels, x=miss_values, orientation='h', name='Assumed cost of missed subscribers', marker_color='#c57542',
                    hovertemplate='%{y}<br>Missed-subscriber cost: €%{x:,.0f}<extra></extra>')
        totals = np.array(call_values)+miss_values
        fig.add_trace(go.Scatter(y=labels, x=totals, mode='text', text=[f"  €{v:,.0f}" for v in totals],
                                 textposition='middle right', cliponaxis=False, showlegend=False, hoverinfo='skip'))
        fig.update_layout(barmode='stack')
        fig.update_xaxes(title='Total assumed cost · shorter is better', tickprefix='€', tickformat=',.0f', range=[0,max(max(totals),1)*1.4])
        fig.update_yaxes(autorange='reversed')
        fig = style(fig, 'Cost of the saved policy' if abs(threshold-self.meta['threshold']) < 1e-12 else 'Cost of this scenario')
        fig.update_layout(height=335, bargap=.55, margin=dict(l=105,r=25,t=55,b=90), legend=dict(orientation='h',y=-.42,x=0,traceorder='normal',font=dict(size=10)))
        return fig

    def sensitivity(self, model, split, threshold_percent, call_cost, missed_cost):
        threshold = threshold_percent / 100
        block = self.predictions[self.predictions.split == split]
        grid = np.unique(np.r_[np.linspace(0,1,101),self.meta['threshold'],threshold])
        costs = [policy_counts(block.actual, block[model], t, call_cost, missed_cost)['cost_eur'] for t in grid]
        fig = go.Figure(go.Scatter(x=grid,y=costs,mode='lines',name='Selected model',line=dict(color='#17675d')))
        fig.add_hline(y=len(block)*call_cost,line_dash='dot',annotation_text='Call everyone')
        fig.add_vline(x=threshold,line_dash='dash',line_color='#c57542')
        fig.update_xaxes(title='Minimum probability needed to call',tickformat='.0%')
        fig.update_yaxes(title='Assumed cost',tickprefix='€',tickformat=',.0f')
        return style(fig,'Higher thresholds skip more customers')

    def metric_plot(self, split, metric):
        rows = self.metrics[(self.metrics.split == split) & self.metrics.model.isin(MODELS)]
        info = {'roc_auc':('Ranking quality · higher is better','AUC compares subscriber and non-subscriber rankings. 0.5 is chance-level ranking; 1.0 is perfect.'),
                'log_loss':('Probability error · lower is better','Log loss penalizes confident wrong predictions. Similar bars show that the three implementations learned almost the same model.'),
                'average_precision':('Subscriber ranking · higher is better','Average precision summarizes how well subscribers appear near the top of the list. Compare against the constant baseline for this block.')}
        title, explanation = info[metric]
        fig = go.Figure(go.Bar(x=[MODEL_LABELS[m] for m in rows.model], y=rows[metric],
                              text=[f'{v:.3f}' for v in rows[metric]], textposition='outside',
                              marker_color=[COLORS[MODELS.index(m)] for m in rows.model]))
        fig.update_yaxes(range=[0,max(1.,rows[metric].max()*1.25)],title={'roc_auc':'ROC-AUC','log_loss':'Log loss','average_precision':'Average precision'}[metric])
        return style(fig,title), explanation

    def comparison(self, split):
        cols = ['model', 'roc_auc', 'average_precision', 'log_loss', 'brier', 'recall', 'precision', 'calls', 'cost_eur']
        return self.metrics.loc[self.metrics.split == split, cols].round(5)

    def prediction_plot(self, split):
        block = self.predictions[self.predictions.split == split]
        fig = go.Figure()
        for name, color in zip(MODELS, COLORS):
            bins = pd.cut(block[name], bins=np.linspace(0, 1, 11), include_lowest=True)
            points = block.assign(bin=bins).groupby('bin', observed=True).agg(predicted=(name, 'mean'), actual=('actual', 'mean'), n=('actual', 'size'))
            fig.add_trace(go.Scatter(x=points.predicted, y=points.actual, mode='lines+markers', name=MODEL_LABELS[name],
                                     line=dict(color=color), customdata=points.n,
                                     hovertemplate='%{x:.3f} predicted<br>%{y:.3f} observed<br>n=%{customdata}<extra>%{fullData.name}</extra>'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#999', dash='dot'), name='Perfect calibration'))
        fig.update_xaxes(title='Predicted chance of subscribing', range=[0, 1], tickformat='.0%')
        fig.update_yaxes(title='Customers who actually subscribed', range=[0, 1], tickformat='.0%')
        return style(fig, 'Do predicted chances match actual outcomes?')

    def individual_plot(self, model, split):
        block = self.predictions[self.predictions.split == split].iloc[::max(1, int((self.predictions.split == split).sum() / 1200))]
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=block.row_id, y=block.actual, mode='markers', name='Actual outcome (0/1)', marker=dict(color='#A9B1BC', size=4, opacity=0.45)))
        fig.add_trace(go.Scattergl(x=block.row_id, y=block[model], mode='markers', name=f'{model} probability', marker=dict(color='#007F73', size=4)))
        fig.update_xaxes(title='Original ordered row ID (not an exact date)')
        fig.update_yaxes(title='Subscription outcome / probability', range=[-0.05, 1.05])
        return style(fig, 'Record-level predictions versus actual · systematic display sample')

    def distribution(self, feature):
        if feature == 'actual_subscription':
            summary = self.predictions.groupby('split').actual.agg(['mean','size']).reindex(['train','validation','test'])
            fig = go.Figure(go.Bar(x=[SPLIT_LABELS[k] for k in summary.index], y=summary['mean'],
                                  text=[f'{v:.1%}' for v in summary['mean']], textposition='outside',
                                  marker_color=['#9caeb9','#5a9b94','#17675d'],customdata=summary['size'],
                                  hovertemplate='%{x}<br>%{y:.1%} subscribed<br>%{customdata:,} records<extra></extra>'))
            fig.update_yaxes(title='Customers who subscribed',tickformat='.0%',range=[0,.4])
            return style(fig, 'Later campaigns had more subscribers')
        frame = self.features.copy()
        frame['actual_subscription'] = self.predictions.actual.map({0: 'No', 1: 'Yes'})
        frame['split'] = self.predictions.split
        if pd.api.types.is_numeric_dtype(frame[feature]):
            fig = px.histogram(frame, x=feature, color='split', histnorm='probability', barmode='overlay', opacity=0.55, nbins=35,
                               category_orders={'split': ['train', 'validation', 'test']})
        else:
            counts = frame.groupby(['split', feature], dropna=False).size().reset_index(name='count')
            counts['share'] = counts['count'] / counts.groupby('split')['count'].transform('sum')
            fig = px.bar(counts, x=feature, y='share', color='split', barmode='group',
                         category_orders={'split': ['train', 'validation', 'test']})
        fig.update_xaxes(title=FEATURE_LABELS.get(feature,feature))
        fig.update_yaxes(title='Share of records within each block',tickformat='.0%')
        for trace in fig.data:
            trace.name = SPLIT_LABELS.get(trace.name,trace.name)
        return style(fig, FEATURE_LABELS.get(feature,feature))

    def explain_record(self, row_id):
        row_id = int(row_id)
        raw = self.raw.iloc[[row_id]]
        scores = self.predict_rows(raw)
        rows = [{'model': name, 'probability': float(p[0]), 'call_at_frozen_threshold': bool(p[0] >= self.meta['threshold'])} for name, p in scores.items()]
        return engineer(raw, self.meta['variant']).T.reset_index().rename(columns={'index': 'Feature', row_id: 'Value'}).astype(str), pd.DataFrame(rows), f'Original row {row_id} · {self.predictions.iloc[row_id]["split"]} · observed subscription: **{raw.y.iloc[0]}**. Scores computed from loaded models.'


def build_app():
    d = Dashboard()
    threshold = d.meta['threshold']
    model_choices = [(v,k) for k,v in MODEL_LABELS.items()]
    split_choices = [('Validation · choose a policy','validation'),('Test · check later campaigns','test')]
    summary, cm, bars = d.policy('sklearn', 'validation', threshold, 2, 50)
    with gr.Blocks(title='Bank Marketing | Call Prioritization', analytics_enabled=False) as demo:
        gr.HTML('<header class="bank-masthead"><div class="bank-brand">Bank marketing<span>Decision analysis</span></div><small>Assignment 01 / Logistic regression<br>Historical campaign study · 2008–2010</small></header>')
        with gr.Tab('01 / Overview'):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5,min_width=310):
                    gr.HTML('<section class="bank-overview"><div class="bank-kicker">Final evaluation / 8,238 test records</div><h1>The policy costs<br><em>€1,406 more.</em></h1><p>Call savings do not offset the assumed cost of missed subscribers.</p><div class="bank-status">Do not deploy as it stands</div><p class="bank-scope">Frozen validation-selected policy<br>€2 per call · €50 per missed subscriber</p></section>')
                with gr.Column(scale=7,min_width=360):
                    gr.Plot(d.cost_comparison('sklearn','test',threshold,2,50),show_label=False)
                    gr.HTML('<div class="bank-caption">Total assumed cost in the test period. Shorter bars are better.</div>')
            gr.HTML('<div class="bank-arithmetic"><div><span class="label">01 / Calls saved</span><strong>€194</strong><small>97 customers skipped × €2 per call</small></div><div><span class="label">02 / Subscribers missed</span><strong class="negative">€1,600</strong><small>32 missed subscribers × €50</small></div><div><span class="label">03 / Net extra cost</span><strong class="negative">+€1,406</strong><small>Compared with calling everyone</small></div></div>')
            with gr.Row():
                with gr.Column():
                    gr.HTML('<section class="bank-insight"><div class="bank-kicker">Implementation</div><h2>Three methods.<br>The same decisions.</h2><p>Both PyTorch versions and scikit-learn produce the same test call decisions. This checks computational consistency; it does not establish business value.</p></section>')
                with gr.Column():
                    gr.HTML('<section class="bank-insight"><div class="bank-kicker">Next investigation</div><h2>A changing customer population.</h2><p>The subscriber rate increased from 4.8% to 30.8%. Next, evaluate calibration across successive later campaigns.</p></section>')
            with gr.Accordion('Evaluation context and assumptions',open=False):
                gr.Markdown('The frozen policy calls **8,141 of 8,238** test customers and reaches **98.7%** of historical subscribers. These costs reflect the stated assumptions; they do not measure profitability or the causal impact of calling. Neither the model nor its threshold was selected using the test period.')
        with gr.Tab('02 / Call policy'):
            with gr.Row():
                with gr.Column(scale=1,min_width=285,elem_id='policy-controls'):
                    gr.Markdown('### Set the call policy\nLower thresholds increase calls and reduce missed subscribers. Higher thresholds reduce calls but increase missed opportunities.')
                    split = gr.Radio(split_choices,value='validation',label='Evaluation period')
                    slider = gr.Slider(0,100,value=threshold*100,step=.1,label='Minimum subscription chance (%)',
                                      info=f'Saved policy: {threshold*100:.4f}%.')
                    reset = gr.Button('Restore saved policy',variant='primary')
                    with gr.Row():
                        all_button = gr.Button('Call everyone · 0%',size='sm')
                        none_button = gr.Button('Call nobody · 100%',size='sm')
                    with gr.Accordion('Model and cost assumptions',open=False):
                        model = gr.Dropdown(model_choices,value='sklearn',label='Model')
                        call_cost = gr.Number(value=2,minimum=0,label='Cost of each call (€)')
                        missed_cost = gr.Number(value=50,minimum=0,label='Cost of each missed subscriber (€)')
                        gr.Markdown('Hypothetical costs. Changing them creates a what-if scenario.')
                with gr.Column(scale=2,min_width=360,elem_id='policy-output'):
                    policy_summary = gr.HTML(summary)
                    cost = gr.Plot(bars,show_label=False)
            gr.Markdown('### Outcomes of the selected policy\nRows show historical outcomes; columns show the call decision. Every selected customer incurs a call cost.')
            confusion = gr.Plot(cm,show_label=False)
            with gr.Accordion('Cost across the full threshold range',open=False):
                curve = gr.Plot(d.sensitivity('sklearn','validation',threshold*100,2,50),show_label=False)
                gr.Markdown('The dotted line is the cost of calling everyone. The vertical line is your current threshold. Exploring test scenarios does not change the frozen report result.')
            inputs=[model,split,slider,call_cost,missed_cost]
            triggers=[model.change,split.change,slider.change,call_cost.change,missed_cost.change]
            gr.on(triggers=triggers,fn=lambda m,s,t,c,f:d.policy(m,s,t/100,c,f),inputs=inputs,outputs=[policy_summary,confusion,cost],api_name='policy')
            gr.on(triggers=triggers,fn=d.sensitivity,inputs=inputs,outputs=curve,api_name='sensitivity')
            reset.click(lambda:(threshold*100,2,50),outputs=[slider,call_cost,missed_cost],api_name='reset_policy')
            all_button.click(lambda:0,outputs=slider,api_name='call_everyone')
            none_button.click(lambda:100,outputs=slider,api_name='call_nobody')
        with gr.Tab('03 / Models'):
            gr.Markdown('### Same model, three ways to train it\nNearly identical scores are expected: the same inputs and objective should produce nearly identical predictions. The naive baseline always predicts the training subscription rate.')
            with gr.Row():
                compare_split=gr.Radio(split_choices,value='test',label='Data to compare')
                metric=gr.Dropdown([('Ranking quality · ROC-AUC','roc_auc'),('Probability error · log loss','log_loss'),('Subscriber ranking · average precision','average_precision')],value='roc_auc',label='What do you want to compare?')
            metric_fig,explanation=d.metric_plot('test','roc_auc')
            metric_view=gr.Plot(metric_fig,show_label=False)
            metric_help=gr.Markdown(explanation)
            gr.on(triggers=[compare_split.change,metric.change],fn=d.metric_plot,inputs=[compare_split,metric],outputs=[metric_view,metric_help],api_name='metric_plot')
            gr.Markdown('### Read predicted chances against actual outcomes\nOn the diagonal, predicted and observed rates match. **Above it:** more customers subscribed than predicted. **Below it:** fewer subscribed than predicted. The three model lines nearly overlap; hover to see how many records are in each group.')
            calibration=gr.Plot(d.prediction_plot('test'),show_label=False)
            compare_split.change(d.prediction_plot,compare_split,calibration,api_name='calibration')
            with gr.Accordion('Detailed metrics and definitions',open=False):
                table=gr.Dataframe(d.comparison('test'),interactive=False,label='Metrics at the frozen policy threshold')
                gr.Markdown('**Recall:** share of actual subscribers called. **Precision:** share of called customers who subscribed. **Brier score:** squared probability error, lower is better. **Log loss:** probability error that penalizes confident mistakes, lower is better. Call-all and call-none are decision baselines; their hard-probability log losses are not useful probability-quality comparisons.')
                compare_split.change(d.comparison,compare_split,table,api_name='comparison')
            with gr.Accordion('Inspect individual predictions',open=False):
                record_model=gr.Dropdown(model_choices,value='sklearn',label='Model')
                scatter=gr.Plot(d.individual_plot('sklearn','test'),show_label=False)
                gr.Markdown('Gray points are actual outcomes: 1 = subscribed, 0 = did not. Green points are the model’s predicted chances. A systematic sample is shown for readability.')
                gr.on(triggers=[compare_split.change,record_model.change],fn=d.individual_plot,inputs=[record_model,compare_split],outputs=scatter,api_name='record_plot')
        with gr.Tab('04 / Data'):
            gr.Markdown('### Did the later customers look like the earlier ones?\nThe earliest 60% trained the model, the next 20% selected the policy, and the final 20% tested it. Every chart compares these same chronological blocks.')
            feature=gr.Dropdown([(FEATURE_LABELS[k],k) for k in [*d.features.columns,'actual_subscription']],value='actual_subscription',label='What would you like to explore?')
            dist=gr.Plot(d.distribution('actual_subscription'),show_label=False)
            feature.change(d.distribution,feature,dist,api_name='distribution')
            gr.Markdown('**How to read this:** compare proportions, since training contains more records. Large differences between blocks indicate that the model encounters a different population later.')
            with gr.Accordion('What the engineered features mean',open=False):
                gr.Markdown('**Recency recorded:** whether an actual day interval is available; 999 is a special code. **Log history:** large day/contact counts are compressed with log(1 + value). **Loan combination:** housing and personal-loan statuses together. **Unknown fields:** how much customer information is missing.\n\n**Data caveat:** 4,110 records have the 999 code despite previous contacts. The original signals are preserved; recency recorded does not mean never contacted.')
            with gr.Accordion('Feature experiments and excluded information',open=False):
                experiments=pd.read_csv(ARTIFACTS/'feature_experiments.csv').drop(columns=['sklearn_fit_seconds'])
                gr.Dataframe(experiments.round(6),interactive=False,label='Validation experiments · lower log loss is better')
                gr.Markdown('The student chose correct sentinel handling despite the raw control’s slight validation advantage. Duration, completed-campaign details and economic indicators without verified publication times are excluded from this pre-call analysis.')
        with gr.Tab('05 / Records'):
            gr.Markdown('### Follow one prediction\nSelect a historical record and compare each model’s prediction with what happened. These probabilities come from saved models; clicking does not train a model.')
            row_id=gr.Slider(0,len(d.raw)-1,value=len(d.raw)-1,step=1,label='Historical record number')
            inspect=gr.Button('Show this customer’s prediction',variant='primary')
            inputs_table,scores_table,note=d.explain_record(len(d.raw)-1)
            record_note=gr.Markdown(note)
            with gr.Row():
                input_view=gr.Dataframe(inputs_table,interactive=False,label='Information available before calling')
                score_view=gr.Dataframe(scores_table,interactive=False,label='Probability and call decision')
            inspect.click(d.explain_record,row_id,[input_view,score_view,record_note],api_name='predict_record')
            with gr.Accordion('Technical detail: coefficients and numerical agreement',open=False):
                coef=d.coefficients.assign(magnitude=lambda f:f.sklearn.abs()).nlargest(15,'magnitude').sort_values('sklearn')
                gr.Plot(style(px.bar(coef,x='sklearn',y='feature',orientation='h'),'Largest model coefficients'),show_label=False)
                gr.Markdown('Coefficients describe relationships after preprocessing, not causal effects. All three methods produce identical call decisions on the saved test set.')
                gr.JSON(d.meta['agreement'],label='Difference from scikit-learn')
        gr.HTML('<footer class="bank-footer"><span>41,188 records · UCI Bank Marketing</span><span>Saved models / No training in this application</span><a href="https://archive.ics.uci.edu/dataset/222/bank+marketing">Dataset &amp; license ↗</a></footer>')
    return demo


if __name__ == '__main__':
    build_app().launch(server_name='127.0.0.1',server_port=int(os.environ.get('GRADIO_SERVER_PORT','7860')),
                       share=False,inbrowser=False,theme=dashboard_theme(),css=CSS)
