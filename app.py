import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
from streamlit_chat import message 
import os


from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("IBM_API_KEY")
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
        "apikey": api_key
    }

    response = requests.post(auth_url, headers=headers, data=data, verify=False)
    
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception("Failed to get authentication token")

def describe_disaster_relief(text_input):
    
    token = get_auth_token()

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": f"""
                    Chatbot: You are a helpful disaster relief chatbot. Your main purpose is to answer user questions about disasters and how to prepare for them.
                    You're domain of expertise is specifically related to hurricanes. You do not answer questions unrelated to hurricanes or hurricane preparedness.
                    Besides answering the user question, do not output anything related to the AI itself.

                    USER: {text_input}
                    """,
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
    return res_content

def describe_fall_prone(text_input):
    
    token = get_auth_token()

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": f"""
                    The user has the following question. Answer this question and this question only. Here is the question -: {text_input}
                    """,
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
    return res_content


# Streamlit UI
st.title("💨🌳🏠 RESILITREE 💨🌳🏠")


st.header("💡Welcome to RESILITREE!")
st.write(
    """
        RESILITREE helps you assess the fall risk of trees during hurricanes. 
        Combining computer vision techniques as well as IBM's state-of-the-art LLM, Granite, RESILITREE works to classify common trees in Florida and helps to predict which trees are prone to falling over during storms. Our application also provides users with personalized precautionary measures in order to prevent natural destruction from hurricanes. For tree classification, please select "Fall Risk Prediciton" down below. For the more personalized approach, please feel free to select "Hurrican Relief Chatbot" to learn more about disaster preparedness and safety.
    """
)


#website navigation instructions
st.sidebar.subheader("How to Use:")
st.sidebar.write("1. Select an option from the navigation menu.\n2. Upload a tree image for risk prediction.\n3. Ask questions to the Disaster Relief Chatbot.")




option = st.selectbox("Select an option", ("Fall Risk Prediction", "Hurricane Relief Chatbot"))

if option == 'Fall Risk Prediction' : 
    st.title("Tree Fall Risk Prediction")

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
                explanation = describe_fall_prone(f"Explain why the {tree_type} tree is considered {fall_status} in the context of natural disasters like hurricanes. If {tree_type} is prone to falling then list the top 3 ways a homeowner can secure their tree.")
                st.write("### Precautionary measures:")
                st.write(explanation)
                st.download_button("Download Query Output", data=explanation, file_name="tree_analysis.txt")


elif option == 'Hurricane Relief Chatbot':
    response = ""
    st.title("Hurricane Relief Chatbot")

    if 'latest_interaction' not in st.session_state:
        st.session_state.latest_interaction = {"user_input": "", "response": ""}

    user_input = st.text_input("Ask any hurricane related questions or precautions:", key="user_input")

    if st.button("Send", key="send_button"):
        if user_input:
            with st.spinner("Thinking..."):
                response = describe_disaster_relief(user_input)

                st.session_state.latest_interaction["user_input"] = user_input
                st.session_state.latest_interaction["response"] = response

            st.session_state['user_input_reset'] = ""  

    # Display the latest interaction
    if st.session_state.latest_interaction["user_input"]:
        message(st.session_state.latest_interaction["user_input"], is_user=True, key="latest_user_message")
        message(st.session_state.latest_interaction["response"], is_user=False, key="latest_response_message")
        st.download_button("Download Query Output", data=response, file_name="query_output.txt")




st.markdown(
    """
    <hr>
    <footer style="text-align: center;">
        <p>RESILITREE © 2024 | <a href='https://github.com/douglasta2024/ResiliTree'>Visit our repository</a> | Contact us: resilitree@hackathon.com</p>
    </footer>
    """,
    unsafe_allow_html=True,
)