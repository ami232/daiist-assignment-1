import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. LOAD ARTIFACTS (Zero Retraining)
# ==========================================
# Load the saved models, scaler, and test artifacts directly from disk
sklearn_model = joblib.load("models/sklearn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
test_artifacts = joblib.load("models/test_artifacts.pkl")

X_test = test_artifacts['X_test']
y_test = test_artifacts['y_test']
sklearn_probs = test_artifacts['sklearn_probs'] # un paquete con las probabilidades de prueba de todos los modelos
manual_probs = test_artifacts['manual_probs']
std_probs = test_artifacts['std_probs']

# Re-define the Standard PyTorch model class to load its weights properly
class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim): #número de columnas o características
        super().__init__() #para iniciar librería
        self.linear = nn.Linear(input_dim, 1) #hace la operación matemática , cada entrada por su peso + el sesgo. Devuelve la única salida que nos interesa que es 1
    def forward(self, x):
        return torch.sigmoid(self.linear(x)) #número entre 0 y 1

n_features = X_test.shape[1] #estamos guardando cuántas columnas tiene nuestro modelo. Esto se hace para que la red sepa exactamente cuántes entradas va a recibir sin tener que escribir el número a mano 
std_model = LogisticRegressionModule(n_features)
std_model.load_state_dict(torch.load("models/std_pytorch.pt")) #para no entrenar la red desde cero, cogemos los pesos guardados y se lo inyectamos al modelo
std_model.eval()


# ==========================================
# 2. GRADIO INTERFACE FUNCTIONS
# ==========================================

# A) Individual customer prediction
def predict_single_customer(tenure, monthly_charges, total_charges):
    """Takes customer data, scales it, and predicts churn across all 3 models."""
    # Create a dummy DataFrame matching the expected training structure
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, total_charges / (tenure + 1)]], 
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerMonthTenure'])
    
    # Fill missing columns with zeros to match the original dataset shape
    for col in X_test.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X_test.columns] # Reorder columns
    
    # Scale input data using the exact scaler fitted on training data
    input_scaled = scaler.transform(input_data) #solo transform para evitar data leakage , misma normalización y desviación típica
    
    # 1. Scikit-Learn prediction probability
    sk_prob = sklearn_model.predict_proba(input_scaled)[0][1] # primera fila , segunda columna que es la probabilidad de churn (irse)
    
    # 2. Manual PyTorch prediction probability (loading weights from disk)
    manual_weights = torch.load("models/manual_pytorch.pt")
    tensor_in = torch.tensor(input_scaled, dtype=torch.float32) #formato exigido para operar
    man_prob = torch.sigmoid(torch.matmul(tensor_in, manual_weights['w']) + manual_weights['b']).item() #extrae el número decimal puro del tensor de Pytorch 
    #lo de sigmoid es para obtener la probabilidad
    
    # 3. Standard PyTorch prediction probability
    with torch.no_grad(): #desactiva el cálculo de gradientes 
        std_prob = std_model(tensor_in).item()#extrae el valor numérico decimal
        
    return f"Scikit-Learn: {sk_prob:.2f} | Manual PyTorch: {man_prob:.2f} | Standard PyTorch: {std_prob:.2f}"


# B) Threshold Analysis, Confusion Matrix, and Business Costs
def evaluate_threshold(model_choice, threshold): #umbral
    """Updates the decision threshold, recalculates the confusion matrix, and estimates business cost."""
    # Select probability array based on chosen model
    if model_choice == "Scikit-Learn":
        probs = sklearn_probs
    elif model_choice == "PyTorch Manual":
        probs = manual_probs
    else:
        probs = std_probs
        
    # Convert probabilities to binary predictions (0 or 1) based on the threshold
    preds = (probs >= threshold).astype(int) #convierte los booleans en números
    
    # Compute confusion matrix (True Negatives, False Positives, False Negatives, True Positives)
    cm = confusion_matrix(y_test, preds) #comprueba respuestas reales (y_test) con predicciones
    tn, fp, fn, tp = cm.ravel() #aplana la mariz para guardar los elementos en 4 variables
    
    # Business cost framing: Prioritizing high-recall to minimize False Negatives (lost revenue from churn)
    # - False Negative (Customer leaves and we do nothing): High cost (e.g., $200 lost revenue)
    # - False Positive (We offer a retention discount to a loyal customer): Low cost (e.g., $50 promo cost)
    cost_fn = 200
    cost_fp = 50
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    output_text = f"""
    ### 📊 Results for Threshold: {threshold:.2f}
    - **True Negatives (TN):** {tn}
    - **False Positives (FP - False Alarm):** {fp}
    - **False Negatives (FN - Missed Churners):** {fn} ⚠️ *(Priority: minimize this)*
    - **True Positives (TP - Caught Churners):** {tp}
    
    ---
    💰 **Total Estimated Business Cost:** **${total_cost}**
    *(Calculated penalizing each False Negative by ${cost_fn} and each False Positive by ${cost_fp})*
    """
    return output_text


# ==========================================
# 3. BUILD GRADIO DASHBOARD
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Telco Churn Prediction Dashboard") #para crear titulo 
    gr.Markdown("Interactive application to evaluate Machine Learning and Deep Learning models without retraining.")

    with gr.Tabs(): #organiza para crear secciones limpias y no saturar la pantalla en pestañas 
        # TAB 1: Individual Prediction
        with gr.TabItem("👤 Single Customer Prediction"): #primera pestaña
            gr.Markdown("Enter customer details to see predictions from all three models:")
            with gr.Row(): #organiza elementos en una fila horizontal
                tenure_input = gr.Number(label="Tenure (Months)", value=12)
                monthly_input = gr.Number(label="Monthly Charges ($)", value=70.0)
                total_input = gr.Number(label="Total Charges ($)", value=850.0)
            
            predict_btn = gr.Button("Predict Churn", variant="primary")
            output_result = gr.Textbox(label="Churn Probability (0 = Stays, 1 = Leaves)")
            
            predict_btn.click(predict_single_customer, inputs=[tenure_input, monthly_input, total_input], outputs=output_result) #conecta a interactividad del botón 

        # TAB 2: Threshold & Business Cost Analysis
        with gr.TabItem("⚙️ Threshold & Business Costs"):
            gr.Markdown("Move the slider to change the decision threshold and observe its impact on classification errors and financial costs.")
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=["Scikit-Learn", "PyTorch Manual", "PyTorch Estándar"], 
                                             value="Scikit-Learn", label="Select Model") # crea un menú despegable 
                threshold_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, 
                                             label="Decision Threshold")
            
            cost_output = gr.Markdown()
            
            threshold_slider.change(evaluate_threshold, inputs=[model_dropdown, threshold_slider], outputs=cost_output)
            model_dropdown.change(evaluate_threshold, inputs=[model_dropdown, threshold_slider], outputs=cost_output)
            # indica que tanto si se mueve el deslizador o se cambia de modelo , se debe ejecutar automáticamente la función evaluate con esos parámetros

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())