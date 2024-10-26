from ibm_watson import IAMTokenManager
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
from streamlit_chat import message 

model = models.efficientnet_b0(pretrained=False)  # Example: Adjust to match your architecture

# Ensure the output features match your number of classes (e.g., 6 classes for 6 tree types)
num_classes = 6
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Step 2: Load the state dictionary into the model
model.load_state_dict(torch.load('tree_detection_model_optimized.pth', map_location=torch.device('cpu')))
model.eval() 

# Define tree type lists
fall_prone_trees = ["Maple", "Eucalyptus", "Queen Palm"]
fall_not_prone_trees = ['Live Oak', 'South Magnolia', 'Sabel Palm']


# Define the transformations (ensure these match the ones used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_tree_type(img):
    # Preprocess the image
    image = Image.open(img).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Check if GPU is available and move the tensor to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        print(outputs)
        _, predicted_class = torch.max((outputs),1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()

    # Define class names (ensure these match your model training)
    class_names = ["Eucalyptus", "Live Oak", "Maple", "Queen Palm", "Sabel Palm", "South Magnolia"] 
    print("Pred", predicted_class) 
    return class_names[predicted_class], confidence

def check_fall_prone(tree_type):
    if tree_type in fall_prone_trees:
        return True, "fall-prone"
    elif tree_type in fall_not_prone_trees:
        return False, "not fall-prone"
    else:
        return None, "unknown" 
    

def get_auth_token():
    
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": 'IDZqAzLSzXq70oDqxI8pAEOheMKKjLTL5TIx0lyt-QJN'
    }

    response = requests.post(auth_url, headers=headers, data=data, verify=False)
    
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception("Failed to get authentication token")


def describe_fall_prone(text_input):
    
    token = get_auth_token()

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": f"{text_input}",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 900,
            "repetition_penalty": 1.05
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": "c402eca2-1dc4-4829-b51e-f5caf31f8676"
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )
    print(response.text)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    res_content = data['results'][0]['generated_text']
    return res_content.lstrip('.\n')


# Streamlit UI
st.title("Tree Type Classification and Fall Risk Detection")
uploaded_file = st.file_uploader("Upload an image of a tree", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Tree Image", use_column_width=True)

    if st.button("Tell me the type of the tree"):
        # Predict tree type
        tree_type, confidence = predict_tree_type(uploaded_file)
        st.write(f"Predicted Tree Type: **{tree_type}** with confidence {confidence * 100:.2f}%")

        # Check if the tree is fall-prone
        is_fall_prone, fall_status = check_fall_prone(tree_type)
        if fall_status == "unknown":
            st.write(f"The tree type **{tree_type}** is not recognized in the current fall-prone or not fall-prone lists.")
        else:
            st.write(f"The tree is categorized as **{fall_status}**.")

            # Provide explanation using an LLM
            explanation = describe_fall_prone(f"Explain why the {tree_type} tree is considered {fall_status} in the context of natural disasters like hurricanes. If the tree is prone to falling, have a separate heading and give suggestions on how to keep {tree_type} stabilize during hurricanes, specifically suggestions that a person can do last minute to prevent their tree from falling. If the tree is not prone to falling then state why {tree_type} does well in hurricanes and give a list of suggestions on how a homeowner could secure their home instead.")
            st.write("### Precautionary measures:")
            st.write(explanation)

import streamlit as st
from streamlit_chat import message

# Chatbot Interface in Sidebar
st.sidebar.title("Disaster and Tree Precautionary Chatbot")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User message input in the sidebar
user_input = st.sidebar.text_input("Ask about disasters or precautions:", key="user_input")

if st.sidebar.button("Send", key="send_button"):
    if user_input:
        with st.spinner("Thinking..."):
            response = describe_fall_prone(user_input)
        # Append user input and response to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Chatbot", response))

# Display chat history in the sidebar
for i, (speaker, message_content) in enumerate(st.session_state.chat_history):
    with st.sidebar:
        message(message_content, is_user=(speaker == "You"), key=f"message_{i}")

