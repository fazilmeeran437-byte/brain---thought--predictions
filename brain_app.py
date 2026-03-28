import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pyttsx3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- VOICE ENGINE ----------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------------- 3D BRAIN ANIMATION ----------------
def brain_3d_animation():

    st.subheader("🧠 3D Brain Activity")

    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 50)

    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones(50), np.cos(phi))

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

    fig.update_layout(
        title="3D Brain Simulation",
        margin=dict(l=0,r=0,b=0,t=30)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- NEURAL NETWORK ANIMATION ----------------
def neural_network_animation():

    st.subheader("⚡ Neural Network Activity")

    fig, ax = plt.subplots()

    # nodes
    layer1 = [1,2,3,4]
    layer2 = [1.5,2.5,3.5]
    layer3 = [2.5]

    for n in layer1:
        ax.scatter(1,n,s=500)

    for n in layer2:
        ax.scatter(2,n,s=500)

    for n in layer3:
        ax.scatter(3,n,s=500)

    # connections
    for i in layer1:
        for j in layer2:
            ax.plot([1,2],[i,j])

    for j in layer2:
        for k in layer3:
            ax.plot([2,3],[j,k])

    ax.axis("off")

    st.pyplot(fig)

# ---------------- STREAMLIT APP ----------------

st.title("🧠 Brain Thought Prediction using EEG")

# Load Dataset
data = pd.read_csv("eeg_dataset_5000.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# Input / Output
X = data[['alpha','beta','theta','delta']]
y = data['label']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.matshow(cm)

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# EEG Input
st.subheader("Enter EEG Signal Values")

alpha = st.number_input("Alpha Value", value=10.0)
beta = st.number_input("Beta Value", value=20.0)
theta = st.number_input("Theta Value", value=5.0)
delta = st.number_input("Delta Value", value=2.0)

# Prediction Button
if st.button("🧠 Predict Thought"):

    # 3D Brain Animation
    brain_3d_animation()

    # Neural Network Animation
    neural_network_animation()

    input_data = [[alpha, beta, theta, delta]]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"You are thinking about: {prediction}")

    # Voice Output
    speak(f"You are thinking about {prediction}")

    # Image Display
    try:
        img = Image.open(f"{prediction}.jpg")
        st.image(img, width=300)
    except:
        st.warning("Image not found")