import streamlit as st
import pandas as pd
import os
import base64
import google.generativeai as genai
import time

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
    .prompt-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.title("🤖 Emotional Chatbot")
    
    # Remove API Key Input since it's hardcoded
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
        "Enter a system prompt to guide the model's behavior:",
        "You are an emotional and empathetic assistant that helps users with their questions."
    )
    
    # About section
    st.subheader("About")
    st.markdown("""
    This app allows you to interact with Google's language models using different industry-specific prompts.
    
    1. Upload a CSV file with prompts
    2. Choose a model
    3. Select a prompt from the uploaded file
    4. Start chatting!
    """)

# Main content area
st.title("Emotional Chatbot")
st.subheader("Select a prompt and start chatting with AI models")

# Function to load CSV
@st.cache_data
def load_prompts_from_csv(file):
    df = pd.read_csv(file)
    
    # Define column mappings (accepting both formats)
    required_columns_options = [
        # Original format
        ["IndustryCategoryShort", "Description", "Prompt", "Prior Instructions"],
        # Your format from the screenshot
        ["Industry", "Category", "Short Description", "Prompt", "Prior Instructions"]
    ]
    
    # Check if any of the column formats match
    format_found = False
    for required_columns in required_columns_options:
        if all(col in df.columns for col in required_columns):
            format_found = True
            # If using the screenshot format, rename columns to match expected format
            if required_columns == ["Industry", "Category", "Short Description", "Prompt", "Prior Instructions"]:
                df["IndustryCategoryShort"] = df["Industry"] + " - " + df["Category"]
                df["Description"] = df["Short Description"]
            break
    
    if not format_found:
        st.error("CSV file must contain either IndustryCategoryShort/Description or Industry/Category/Short Description columns")
        return None
    
    return df

# Function to generate response
def generate_response(prompt, user_input, model):
    try:
        # Format the prompt with user input
        formatted_prompt = prompt.replace("INSERT_INPUT_HERE", user_input)
        
        model_instance = genai.GenerativeModel(model_name=model)
        
        # Create a system instruction if provided
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }
        
        if system_prompt:
            convo = model_instance.start_chat(
                history=[],
                system_instruction=system_prompt
            )
            response = convo.send_message(
                formatted_prompt,
                generation_config=generation_config,
                stream=True
            )
        else:
            response = model_instance.generate_content(
                formatted_prompt,
                generation_config=generation_config,
                stream=True
            )
        
        # Stream the response
        response_text = ""
        response_placeholder = st.empty()
        
        for chunk in response:
            chunk_text = chunk.text if hasattr(chunk, 'text') else ""
            if chunk_text:
                response_text += chunk_text
                response_placeholder.markdown(response_text + "▌", unsafe_allow_html=True)
                time.sleep(0.01)  # For a more natural typing effect
        
        response_placeholder.markdown(response_text, unsafe_allow_html=True)
        return response_text
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Load and display prompts if file is uploaded
if uploaded_file is not None:
    prompts_df = load_prompts_from_csv(uploaded_file)
    
    if prompts_df is not None:
        # Create columns for the prompt cards
        num_cols = 3
        cols = st.columns(num_cols)
        
        # Display prompts in a grid
        st.subheader("Available Prompts")
        st.markdown("Click on a prompt to use it in the chat")
        
        # Create a container for the prompt cards
        prompt_container = st.container()
        
        with prompt_container:
            for i, (index, row) in enumerate(prompts_df.iterrows()):
                col_idx = i % num_cols
                with cols[col_idx]:
                    # Get the display title (using either format)
                    if "IndustryCategoryShort" in row:
                        display_title = row['IndustryCategoryShort']
                    else:
                        display_title = f"{row['Industry']} - {row['Category']}"
                    
                    # Get the description (using either format)
                    if "Description" in row:
                        description = row['Description']
                    else:
                        description = row['Short Description']
                    
                    with st.expander(f"{display_title}"):
                        st.markdown(f"**Description:** {description}")
                        if st.button(f"Use this prompt", key=f"prompt_{index}"):
                            st.session_state.selected_prompt = {
                                "industry": display_title,
                                "description": description,
                                "prompt": row['Prompt'],
                                "instructions": row['Prior Instructions']
                            }
        
        # Chat interface
        st.markdown("---")
        st.subheader("Chat Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display selected prompt
        if "selected_prompt" in st.session_state:
            st.markdown(f"**Selected Prompt:** {st.session_state.selected_prompt['industry']}")
            st.markdown(f"**Description:** {st.session_state.selected_prompt['description']}")
        
        # Display chat history
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
        
        # User input
        with st.container():
            user_input = st.text_area("Your message:", key="user_input", height=100)
            send_button = st.button("Send")
            
            if send_button and user_input and "selected_prompt" in st.session_state:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display thinking indicator
                with st.spinner("Thinking..."):
                    # Create the full prompt
                    full_prompt = st.session_state.selected_prompt["instructions"]
                    
                    # Generate response
                    assistant_response = generate_response(full_prompt, user_input, model_option)
                    
                    if assistant_response:
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Clear input
                st.experimental_rerun()
            
            elif send_button and "selected_prompt" not in st.session_state:
                st.warning("Please select a prompt first")
    else:
        st.error("Please upload a valid CSV file with the required columns")
else:
    st.info("Please upload a CSV file with prompts to continue")

# Footer
st.markdown("---")
st.markdown("Emotional Chatbot powered by Google's Language Models")
