import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# Set page config
st.set_page_config(
    page_title="Emotional Chatbot with Google Models",
    page_icon="🤖",
    layout="wide",
)

# Set API key directly in the code
os.environ["GOOGLE_API_KEY"] = "AIzaSyCX5Q42LoLMZJ1H6WY6Ja1eso1gx04ZPJg"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.assistant {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

# Sidebar for configuration
with st.sidebar:
    st.title("🤖 Emotional Chatbot")
    
    # API Key status
    st.success("API Key is pre-configured")
    
    # File uploader for CSV
    st.subheader("Upload Prompts")
    uploaded_file = st.file_uploader("Upload CSV file with prompts", type=["csv"])
    
    # Model selection
    st.subheader("Select Model")
    model_option = st.selectbox(
        "Choose a model:",
        ["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "gemini-2.0-flash"]
    )
    
    # System prompt
    st.subheader("System Prompt (Optional)")
    system_prompt = st.text_area(
        "Enter a system prompt:",
        "You are an emotional and empathetic assistant that helps users with their questions."
    )

# Main content area
st.title("Emotional Chatbot")
st.subheader("Select a prompt and start chatting with AI models")

# Function to load CSV
def load_prompts_from_csv(file):
    try:
        df = pd.read_csv(file)
        
        # Check for required columns
        if all(col in df.columns for col in ["Industry", "Category", "Short Description", "Prompt", "Prior Instructions"]):
            # Add combined columns
            df["DisplayTitle"] = df["Industry"] + " - " + df["Category"]
            return df
        else:
            st.error("CSV must contain: Industry, Category, Short Description, Prompt, Prior Instructions")
            return None
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

# Function to generate response
def generate_response(prompt, user_input, model):
    try:
        # Format the prompt with user input
        formatted_prompt = prompt.replace("INSERT_INPUT_HERE", user_input)
        
        # Add system prompt if provided
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{formatted_prompt}"
        else:
            combined_prompt = formatted_prompt
        
        # Generate response
        model_instance = genai.GenerativeModel(model_name=model)
        response = model_instance.generate_content(combined_prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating a response."

# Display chat messages
def display_chat():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.container():
            st.markdown(f"""
            <div class="chat-message {role}">
                <div><strong>{'You' if role == 'user' else 'Assistant'}</strong></div>
                <div>{content}</div>
            </div>
            """, unsafe_allow_html=True)

# Display prompts if file is uploaded
if uploaded_file is not None:
    prompts_df = load_prompts_from_csv(uploaded_file)
    
    if prompts_df is not None:
        # Display prompts
        st.subheader("Available Prompts")
        
        # Create columns for display
        num_cols = 3
        cols = st.columns(num_cols)
        
        # Display prompts in cards
        for i, (index, row) in enumerate(prompts_df.iterrows()):
            col_idx = i % num_cols
            with cols[col_idx]:
                with st.expander(f"{row['DisplayTitle']}"):
                    st.markdown(f"**Description:** {row['Short Description']}")
                    if st.button(f"Use this prompt", key=f"btn_{index}"):
                        # Store selected prompt in session state
                        st.session_state.selected_prompt = {
                            "title": row['DisplayTitle'],
                            "description": row['Short Description'],
                            "prompt": row['Prompt'],
                            "instructions": row['Prior Instructions']
                        }
        
        # Chat interface
        st.markdown("---")
        st.subheader("Chat Interface")
        
        # Display selected prompt
        if st.session_state.selected_prompt:
            st.markdown(f"**Selected Prompt:** {st.session_state.selected_prompt['title']}")
            st.markdown(f"**Description:** {st.session_state.selected_prompt['description']}")
            
            # Display chat history
            display_chat()
            
            # User input
            with st.container():
                user_input = st.text_area("Your message:", key="user_input", height=100)
                if st.button("Send"):
                    if user_input:
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        
                        # Get response
                        instructions = st.session_state.selected_prompt["instructions"]
                        
                        with st.spinner("Thinking..."):
                            assistant_response = generate_response(instructions, user_input, model_option)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        
                        # Rerun to update UI (with a safer approach)
                        st.rerun()
                    else:
                        st.warning("Please enter a message.")
        else:
            st.info("Select a prompt from the options above to start chatting.")
    else:
        st.error("Please upload a valid CSV file with the required columns.")
else:
    st.info("Please upload a CSV file with prompts to continue.")
