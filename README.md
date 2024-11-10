
---

# RESILITREE 🌳

Welcome to **RESILITREE** - a powerful application designed to help you assess tree fall risks during natural disasters like hurricanes and provide disaster relief guidance through an interactive chatbot.
![image](https://github.com/user-attachments/assets/d940133d-4c5a-4957-9872-a3b27da1c83c)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
  
## Introduction

**RESILITREE** utilizes advanced AI models to predict which trees are prone to falling during hurricanes. Additionally, it provides personalized precautionary measures and offers a Disaster Relief Chatbot for disaster preparedness and safety guidance.

## Features

- **Fall Risk Prediction**: Upload an image of a tree and receive an AI-based prediction of whether it is fall-prone.
- **Disaster Relief Chatbot**: An interactive chatbot providing safety measures and disaster preparedness information.
- **User-Friendly UI**: A simple and intuitive interface built using Streamlit.
- **Continuous Conversations**: A chatbot that maintains the latest interaction context for seamless communication.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web app interface.
- **PyTorch**: For deep learning model implementation.
- **IBM Watson API**: For generating context-specific responses in the chatbot.
- **PIL**: For image processing.
- **Git & GitHub**: Version control and collaboration.

## Getting Started

### Prerequisites

Before you begin, make sure you have the following installed on your system:

- Python 3.7 or higher
- `pip` package manager

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/douglasta2024/ResiliTree.git
   cd ResiliTree
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   Create a `.env` file or add environment variables manually to your shell. Alternatively, use Streamlit's secrets management as described below.

   **Example `.env` file**:
   ```env
   IBM_API_KEY=your_ibm_api_key_here
   ```

5. **Add your secrets in Streamlit** (optional):
   - Create a file `.streamlit/secrets.toml` with your API key:

   ```toml
   [general]
   api_key = "your_ibm_api_key_here"
   ```

## Configuration

Ensure your project is configured correctly by storing secrets like API keys securely:

1. Use environment variables or a `.env` file.
   (OR)
3. Use `.streamlit/secrets.toml` for deployment with Streamlit.

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Navigate to the app**:
   Open your browser and go to `http://localhost:8501`.

3. **Select an option**:
   - **Home**: Learn about RESILITREE and its features.
   - **Fall Risk Prediction**: Upload an image of a tree to receive a prediction.
   - **Disaster Relief Chatbot**: Ask questions related to disaster preparedness.

4. **Upload an image** in the "Fall Risk Prediction" section to get a detailed prediction on tree stability and suggested precautionary measures.

5. **Chat with the Disaster Relief Chatbot** in the "Disaster Relief Chatbot" section for helpful disaster preparedness tips.

## Project Structure

```
📂 resilitree
 ┣ 📂 data          # dataset manually curated and annotated.
 ┃ ┗ 📜 eucalyptus
   ┗ 📜 maple
   ┗ 📜 queenpalm
   ┗ 📜 sabel_palm
   ┗ 📜 south_magnolia
   ┗ 📜 live_oak 
 ┣ 📂 myenv              # environment variables
 ┣ 📂 tree-detection-using-pytorch.ipynb               # Classification model training using pytorch
 ┣ 📜 app.py              # Main Streamlit app script
 ┣ 📜 requirements.txt    # Python dependencies
 ┗ 📜 README.md           # Project documentation
 ┗ 📜 .env                # Set up your env variables/secrets in this file
```

## Contributing

Contributions are welcome! If you find any bugs or have feature requests, please open an issue or submit a pull request.

1. **Fork the repository**.
2. **Create a new branch** for your changes.
3. **Commit your changes**.
4. **Push to your branch**.
5. **Create a pull request** to the `main` branch.
