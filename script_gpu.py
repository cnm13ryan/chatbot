# Import All the Required Libraries
import streamlit as st
import torch  # Import PyTorch
import os

# Set PyTorch device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the PyTorch model
model_path = "/Users/nokman/git/llama2/meta_models/llama-2-7b/consolidated.00.pth"
model = torch.load(model_path, map_location=device)
model.to(device)  # Move the model to GPU

# Add a title to your Streamlit Application on Browser
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot with Streamlit")

# Create a Side bar
with st.sidebar:
    st.title("🦙💬 Llama 2 Chatbot")
    st.header("Settings")
    st.subheader("Models and Parameters")

    select_model = st.selectbox("Choose a Llama 2 Model", ['Llama 2 7b'], key='select_model')
    # Initialize your local Llama2 model here based on the selected model
    if select_model == 'Llama 2 7b':
        model_path = "/path/to/llama-2-7b"

    try:
        model = torch.load(model_path, map_location=device)
        model.to(device)  # Move the model to GPU
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

    temperature = st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

# Store the LLM Generated Response
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display the chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear the Chat Messages
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def prepare_input(prompt_input):
    # Implement your code to prepare the input for the model
    prepared_input = {"text": prompt_input}  # Replace with actual code
    return prepared_input

def run_model(prepared_input, model):
    # Implement your code to run the model and get the output
    output = model(prepared_input["text"])  # Replace with actual code
    return output

# Create a Function to generate the Llama 2 Response
def generate_llama2_response(prompt_input):
    # Prepare your input data (this is just a placeholder)
    prepared_input = prepare_input(prompt_input)  # You'll need to implement prepare_input()
    prepared_input = prepared_input.to(device)  # Move input to GPU

    # Run the model (this is just a placeholder)
    output = run_model(prepared_input, model)  # You'll need to implement run_model()
    output = output.cpu().detach().numpy()  # Move output back to CPU and convert to numpy

    return output

# User-Provided Prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a New Response if the last message is not from the assistant
# Initialize response to a default value
response = "Thinking..."

# Generate a New Response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)  # This will overwrite the default value
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
