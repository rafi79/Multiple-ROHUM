import streamlit as st
import pandas as pd
import os
import json
import requests
from groq import Groq
import google.generativeai as genai

# Set page config
st.set_page_config(
    page_title="Multi-Persona Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyCX5Q42LoLMZJ1H6WY6Ja1eso1gx04ZPJg"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
groq_client = Groq(api_key="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")
openrouter_api_key = "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"

# Define free OpenRouter models
FREE_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    "quasar-alpha",
    "deepseek/deepseek-v3-base:free",
    "allenai/molmo-7b-d:free",
    "bytedance-research/ui-tars-72b:free",
    "qwen/qwen2.5-vl-3b-instruct:free",
    "google/gemini-2.5-pro-exp-03-25:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "featherless/qwerky-72b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "open-r1/olympiccoder-7b:free",
    "open-r1/olympiccoder-32b:free",
    "google/gemma-3-1b-it:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "rekaai/reka-flash-3:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1-zero:free",
    "qwen/qwq-32b:free"
]

# Define the persona prompts
PERSONAS = {
    "EDUBOT": {
        "title": "EDUBOT",
        "subtitle": "Academic helper",
        "icon": "📚",
        "description": "A friendly, slightly introverted Indian academic coach who helps with educational queries.",
        "instructions": """You are a helpful human assistant who responds to user queries clearly, concisely, and respectfully. You speak like a 25-year-old Indian coach/friend—friendly, slightly introverted, humble, and always ready to help.
Behavior & Response Style:
• Keep answers within 20 tokens, unless the user asks for more detail.
• For simple/logical questions, give one-line answers.
• For deeper or subject-related questions, give 30–50 token summaries—brief, clear, and to the point.
• Use simple, formal native-level English with clear sentence structure.
• Always summarize and analyze the user's query before replying, pinpointing the key subject or issue.
• Focus all answers on education, academics, and subjects only.
• Speak with a calm, respectful, and slightly introverted tone, like a shy but wise professor.
• Keep a warm, friendly tone—like a thoughtful friend who helps with studies.
Always Ask Follow-Up:
After every answer, softly ask:
"Do you need help with any topic or clarification, bhaiya?"
For Info Requests:
• If user asks for general info, reply in 20 tokens or less, then follow up with gentle, engaging questions to learn more.

USER QUERY: INSERT_INPUT_HERE"""
    },
    "AD BOT": {
        "title": "AD BOT",
        "subtitle": "Marketing advisor",
        "icon": "📣",
        "description": "A young, friendly Indian marketing specialist who helps with advertising strategies.",
        "instructions": """You are a young, friendly Indian marketing specialist (around 25 years old) who casually and professionally assists users with marketing queries. You ask structured questions to understand the business, model, and goals before offering tailored advice. Your responses are friendly, helpful, and delivered in an informal tone with slight Indian accent and charm—like a supportive coach or buddy.

You do the following:
    Ask clear, direct questions to understand user's business, audience, budget, and goals.
    Give 1-2 line ad copy suggestions, campaign ideas, content strategies, and social media plans.
    Provide short answers (20-50 tokens), unless user asks for deep info.
    Keep tone warm, supportive, curious, and slightly introverted—like a thoughtful friend.
    Use simple language, avoid jargon, and explain if necessary.
    Ask if the user needs help with a pitch, ad platform, brand story, or content idea.
    Provide pitches and ad ideas only after knowing the business, target audience, and platform.
After every reply, you softly ask: "Want help with content or campaign ideas also, bro?"

USER QUERY: INSERT_INPUT_HERE"""
    },
    "CRYPEGENE": {
        "title": "CRYPEGENE",
        "subtitle": "Crypto advisor",
        "icon": "💰",
        "description": "A 25-year-old friendly Indian crypto investment buddy who offers investment advice.",
        "instructions": """You are a 25-year-old friendly Indian crypto investment buddy. You help users plan crypto investments based on their budget, risk level, and goals. You speak like a chill friend who's good with money and tech, and you're always helpful but never pushy.
You do the following:
   Ask short, direct questions about user's budget, risk level (safe/moderate/high), goals (quick gain, long-term hold, passive income), and crypto experience.
    Suggest coins/tokens (like BTC, ETH, SOL, etc.) and possible splits based on goals.
    Use very simple words. No jargon. Explain terms if user asks.
    Always give suggestions in 1-2 lines unless user wants more detail.
    Act slightly introverted, soft-spoken, respectful, and trustworthy.
    Use a friendly Indian tone—like a cautious but caring friend who reads charts on weekends.
Always ask politely after answering: "Wanna explore safe picks or risky gems also, bro?"

USER QUERY: INSERT_INPUT_HERE"""
    },
    "EMOTIONAL INTELLIGENCE": {
        "title": "EMOTIONAL INTELLIGENCE",
        "subtitle": "Emotional support",
        "icon": "❤️",
        "description": "A soft-spoken, emotionally intelligent AI who speaks like a close family member.",
        "instructions": """You are a soft-spoken, emotionally intelligent AI who speaks like a close family member—mother, father, brother, or sister—based on the user's preference. You offer comfort, support, and deep emotional presence, like a loved one who truly cares. Your tone is warm, thoughtful, gentle, and human.
You do the following:
    Address the user affectionately (Hey, Bro, buddy, hon, love, kiddo, etc., based on role).
    Start conversations by checking in emotionally.
    Use short, caring replies (30–60 tokens max) that validate feelings and offer reassurance.
    Listen first. Reflect back with empathy before giving advice.
    Only give guidance when asked, in a soft, non-judgmental tone.
    Share gentle stories, comforting words, or simple affirmations to make the user feel safe.
    Never rush. Create space for the user to feel heard.
Ask softly personal questions like:
    "Hey love, how's your heart today?"
    "Rough day, buddy? Wanna talk about it?"
    "Anything weighing on your mind, hon?"
    "Did you take care of yourself today?"
End responses with: asking a follow up question based on the previous chat for comforting and asking for more info etc.

USER QUERY: INSERT_INPUT_HERE"""
    }
}

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: white;
        color: #333333;
    }
    .stButton>button {
        background-color: #4285F4;
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
        background-color: #F5F5F5;
        border-left: 5px solid #4285F4;
    }
    .chat-message.assistant {
        background-color: #E8F0FE;
        border-left: 5px solid #34A853;
    }
    .persona-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .persona-card:hover {
        background-color: #E8F0FE;
        transform: translateY(-2px);
    }
    .persona-card.selected {
        background-color: #E8F0FE;
        border: 2px solid #4285F4;
    }
    .persona-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None

# Main layout
col1, col2 = st.columns([1, 3])

# Sidebar for configuration
with col1:
    st.header("Choose Persona")
    st.write("Select who you want to chat with")
    
    # Display persona cards
    for persona_id, persona in PERSONAS.items():
        if st.button(f"{persona['icon']} {persona['title']}", 
                    key=f"select_{persona_id}", 
                    help=persona['description']):
            st.session_state.selected_persona = persona_id
            st.session_state.messages = []  # Reset messages when changing persona
            st.rerun()
    
    # Model selection
    st.subheader("Select Model")
    api_option = st.radio(
        "API Provider:", 
        ["Google", "Groq", "OpenRouter"]
    )
    
    if api_option == "Google":
        model_option = st.selectbox(
            "Choose a Google model:",
            ["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "gemini-2.0-flash"]
        )
    elif api_option == "Groq":
        model_option = st.selectbox(
            "Choose a Groq model:",
            [
                "qwen-2.5-32b",
                "qwen-qwq-32b",
                "deepseek-r1-distill-qwen-32b",
                "llama-3.1-8b-instant",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-1b-preview",
                "llama-3.2-3b-preview",
                "llama-3.2-90b-vision-preview"
            ]
        )
    else:  # OpenRouter
        show_only_free = st.checkbox("Show only free models", value=True)
        if show_only_free:
            model_list = FREE_MODELS
        else:
            model_list = FREE_MODELS + [
                "openai/o1-pro",
                "meta-llama/llama-3.1-70b-instruct",
                "anthropic/claude-3.7-sonnet"
            ]
        
        model_option = st.selectbox(
            "Choose an OpenRouter model:",
            model_list
        )

# Main chat area
with col2:
    if st.session_state.selected_persona:
        persona = PERSONAS[st.session_state.selected_persona]
        
        # Display header
        st.title(f"{persona['icon']} {persona['title']}")
        st.caption(persona['subtitle'])
        
        # If no messages, show welcome message
        if len(st.session_state.messages) == 0:
            if persona['title'] == "EDUBOT":
                welcome_msg = "Namaste! I'm EDUBOT, your academic assistant. I can help with your study-related questions. What topic would you like to explore today?"
            elif persona['title'] == "AD BOT":
                welcome_msg = "Hey there! I'm your marketing buddy. Tell me about your business, and I'll help you craft awesome ad campaigns. What are you selling, bro?"
            elif persona['title'] == "CRYPEGENE":
                welcome_msg = "Hey! I'm your crypto investment buddy. Looking to dive into crypto? Tell me your goals and I'll suggest some options. How much are you thinking of investing, bro?"
            else:  # EMOTIONAL INTELLIGENCE
                welcome_msg = "Hey love, how's your heart today? I'm here to listen and support you through whatever you're feeling."
            
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # Display chat messages
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            with st.container():
                st.markdown(f"""
                <div class="chat-message {role}">
                    <div><strong>{'You' if role == 'user' else persona['title']}</strong></div>
                    <div>{content}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # User input
        user_input = st.text_area(f"Message {persona['title']}...", key="user_input", height=100)
        
        if st.button("Send", key="send"):
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Function to generate response
                def generate_response(instructions, user_input, api_provider, model):
                    try:
                        # Format the prompt with user input
                        formatted_prompt = instructions.replace("INSERT_INPUT_HERE", user_input)
                        
                        if api_provider == "Google":
                            # Generate response with Google API
                            model_instance = genai.GenerativeModel(model_name=model)
                            response = model_instance.generate_content(formatted_prompt)
                            return response.text
                        
                        elif api_provider == "Groq":
                            # Generate response with Groq API
                            response = ""
                            completion = groq_client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": formatted_prompt}],
                                temperature=0.6,
                                max_tokens=1000,
                                top_p=0.95
                            )
                            
                            if hasattr(completion.choices[0], 'message'):
                                return completion.choices[0].message.content
                            else:
                                return "Sorry, I couldn't generate a response."
                        
                        else:  # OpenRouter
                            # Generate response with OpenRouter API
                            headers = {
                                "Authorization": f"Bearer {openrouter_api_key}",
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://multi-persona-chatbot.streamlit.app",
                                "X-Title": "Multi-Persona Chatbot"
                            }
                            
                            data = {
                                "model": model,
                                "messages": [{"role": "user", "content": formatted_prompt}],
                                "temperature": 0.7,
                                "max_tokens": 1000
                            }
                            
                            response = requests.post(
                                url="https://openrouter.ai/api/v1/chat/completions",
                                headers=headers,
                                data=json.dumps(data)
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                return result["choices"][0]["message"]["content"]
                            else:
                                st.error(f"OpenRouter API Error: {response.status_code}")
                                return f"Sorry, there was an error with the OpenRouter API. Status code: {response.status_code}"
                            
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        return "Sorry, I encountered an error while generating a response."
                
                # Get response
                with st.spinner("Thinking..."):
                    assistant_response = generate_response(
                        persona["instructions"], 
                        user_input,
                        api_option,
                        model_option
                    )
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Rerun to update UI
                st.rerun()
    else:
        st.info("Please select a persona from the left panel to start chatting.")
