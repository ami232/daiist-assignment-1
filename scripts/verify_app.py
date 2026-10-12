"""Exercise the running Gradio API and ensure no model artifact is rewritten."""
import hashlib
import json
import sys
from pathlib import Path
from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]
artifacts = ROOT / 'artifacts'
def hashes():
    return {str(p.relative_to(artifacts)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.rglob('*') if p.is_file()}
before = hashes()
meta = json.loads((artifacts / 'metadata.json').read_text())
client = Client(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:7860', verbose=False)
count = 0
for model in ['sklearn','manual','standard','baseline']:
    out = client.predict(model, 'test', meta['threshold']*100, 2, 50, api_name='/policy')
    assert ('17,882' if model != 'baseline' else '16,476') in out[0]
    count += 1
for threshold, expected in [(0,'16,476'),(100,'127,000')]:
    assert expected in client.predict('sklearn','test',threshold,2,50,api_name='/policy')[0]
    count += 1
for split in ['validation','test']:
    client.predict(split,api_name='/comparison')
    client.predict(split,api_name='/calibration')
    client.predict('standard',split,api_name='/record_plot')
    count += 3
for feature in ['age','job','marital','education','default','housing','loan','poutcome','recency_recorded',
                'log_recency_days','log_previous_contacts','loan_combination','unknown_customer_fields','actual_subscription']:
    client.predict(feature,api_name='/distribution')
    count += 1
for row in [0,24712,32950,41187]:
    out = client.predict(row,api_name='/predict_record')
    assert f'Original row {row}' in out[2]
    count += 1
client.predict(api_name='/reset_policy'); count += 1
for metric in ['roc_auc','average_precision','log_loss']:
    figure, explanation = client.predict('test',metric,api_name='/metric_plot')
    assert explanation
    count += 1
client.predict('sklearn','test',meta['threshold']*100,2,50,api_name='/sensitivity'); count += 1
assert client.predict(api_name='/call_everyone') == 0
assert client.predict(api_name='/call_nobody') == 100
count += 2
assert hashes() == before, 'App changed saved artifacts'
print(f'{count} Gradio API checks passed; all artifact hashes unchanged.')
