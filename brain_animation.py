import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("🧠 Live Brain Signal Animation")

st.write("Simulated EEG Brain Waves")

# graph placeholder
graph = st.empty()

while True:

    # simulated brain waves
    alpha = np.random.uniform(8,13)
    beta = np.random.uniform(13,30)
    theta = np.random.uniform(4,8)
    delta = np.random.uniform(0.5,4)

    signals = [alpha,beta,theta,delta]

    fig, ax = plt.subplots()

    ax.bar(["Alpha","Beta","Theta","Delta"], signals)

    ax.set_ylim(0,35)

    ax.set_title("Real Time Brain Waves")

    graph.pyplot(fig)

    time.sleep(1)