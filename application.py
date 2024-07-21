import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'C:\\Users\\Asus\\OneDrive\\Desktop\\New folder\\MODELS\\decisiontree_model.pkl'
model = joblib.load(model_path)


# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Anemia Prediction')

    # Add a description
    st.write('Enter patient information to predict anemia.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
        sex = st.selectbox("Patient's Sex", ['Female', 'Male'])
        red_pixel = st.slider("% Red Pixel", 0, 100, 50)
        blue_pixel = st.slider("% Blue Pixel", 0, 100, 50)
        green_pixel = st.slider("% Green Pixel", 0, 100, 50)
        Hb = st.slider("Hemoglobin Level (g/dL)", 5, 20, 12)

    # Convert categorical inputs to numerical
    Sex_f = 1 if sex == 'Female' else 0
    Sex_m = 1 - Sex_f

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        '%Red Pixel': [red_pixel],
        '%Blue pixel': [blue_pixel],
        '%Green pixel': [green_pixel],
        'Hb': [Hb],
        'Sex_f': [Sex_f],
        'Sex_m': [Sex_m]
    })

    # Ensure columns are in the same order as during model training
   
    input_data = input_data[expected_columns]
   

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]

                st.write(f'Prediction for Anemia: {"Yes" if prediction[0] == 1 else "No"}')
                st.write(f'Probability of Anemia: {probability:.2f}')

                # Plotting
                fig, axes = plt.subplots(2, 1, figsize=(8, 12))

                # Plot Anemia probability
                sns.barplot(x=['No', 'Yes'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
                axes[0].set_title('Anemia Probability')
                axes[0].set_ylabel('Probability')

                # Plot Hb distribution
                sns.histplot(input_data['Hb'], kde=True, ax=axes[1])
                axes[1].set_title('Hemoglobin Level Distribution')

                # Display the plots
                st.pyplot(fig)

                # Provide recommendations
                if prediction[0] == 1:
                    st.error("The patient is likely to have anemia. Consult a doctor for further evaluation and treatment.")
                else:
                    st.success("The patient is unlikely to have anemia. Continue with regular health check-ups.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
