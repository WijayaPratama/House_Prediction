import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("house_price_model.pkl")

def predict_price(overall_qual, gr_liv_area, garage_cars):
    """
    Predict house price based on:
    - Overall Quality (1-10)
    - Living Area (square feet)
    - Garage Capacity (number of cars)
    """
    
    # Prepare input as 2D array
    input_data = np.array([[overall_qual, gr_liv_area, garage_cars]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return f"Predicted House Price: ${prediction[0]:,.2f}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, step=1, label="Overall Quality (1-10)"),
        gr.Number(label="Living Area (square feet)"),
        gr.Slider(0, 4, step=1, label="Garage Capacity (number of cars)")
    ],
    outputs="text",
    title="🏠 House Price Prediction App",
    description="This app predicts house prices using a trained Linear Regression model."
)

if __name__ == "__main__":
    interface.launch()