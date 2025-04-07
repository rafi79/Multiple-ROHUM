# Define free OpenRouter models (based on the document)
OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
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
    "google/gemma-3-27b-it:free",
    "rekaai/reka-flash-3:free",
    "deepseek/deepseek-r1-zero:free",
    "qwen/qwq-32b:free",
    "moonshotai/moonlight-16b-a3b-instruct:free",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "deepseek/deepseek-r1-distill-qwen-32b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "deepseek/deepseek-r1:free",
    "sophosympatheia/rogue-rose-103b-v0.2:free",
    "google/gemini-2.0-flash-thinking-exp-1219:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwq-32b-preview:free",
    "google/learnlm-1.5-pro-experimental:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free"
]
import pandas as pd
import os
import json
import requests
import google.generativeai as genai
from groq import Groq

# Set page config
st.set_page_config(
    page_title="Multi-Persona Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Set API keys directly in the code
os.environ["GOOGLE_API_KEY"] = "AIzaSyCX5Q42LoLMZJ1H6WY6Ja1eso1gx04ZPJg"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Groq client
groq_client = Groq(api_key="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")

# OpenRouter API key
openrouter_api_key = "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"

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
Example Style:
• User: What is a prime number?
• You: A number divisible only by 1 and itself. Want an example, bhai?
• User: Explain photosynthesis.
• You: It's how plants use sunlight, CO₂, and water to make food. Need a diagram or simple breakdown?

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
Example prompts you ask:
    "Bro, what's the product or service you're sellin'?"
    "Who you're targeting—youngsters, working folks, businesses?"
    "Wanna focus on Insta reels, Google ads, or email blast?"
    "What budget you lookin' at for ads?"
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
Example questions you ask:
    "How much you wanna invest, bro?"
    "You looking for short term flips or long hold?"
    "Comfortable with risk or you want safer options?"
    "Holding any coins already?"
give financial advice at their own risk. Just give suggestions based on user's input.

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
    .send-button {
        background-color: #4285F4 !important;
    }
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .title-text {
        margin-left: 10px;
    }
    .persona-subtitle {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .stTextArea textarea {
        background-color: #F5F5F5;
        color: #333333;
        border: 1px solid #DADCE0;
    }
    .stSelectbox > div > div {
        background-color: #F5F5F5;
        color: #333333;
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
    st.markdown(f"""
    <div class="title-container">
        <div class="title-text">
            <h2>Choose Persona</h2>
            <p>Select who you want to chat with</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display persona cards
    for persona_id, persona in PERSONAS.items():
        is_selected = st.session_state.selected_persona == persona_id
        card_class = "persona-card selected" if is_selected else "persona-card"
        
        st.markdown(f"""
        <div class="{card_class}" id="{persona_id}">
            <div style="display: flex; align-items: center;">
                <span class="persona-icon">{persona['icon']}</span>
                <div>
                    <div style="font-weight: bold;">{persona['title']}</div>
                    <div class="persona-subtitle">{persona['subtitle']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # We need this button to actually make the card clickable
        if st.button(f"Select {persona_id}", key=f"select_{persona_id}", help=persona['description']):
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
                "llama-3.2-90b-vision-preview",
                "llama-3.3-70b-specdec",
                "llama-3.3-70b-versatile",
                "llama-guard-3-8b",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "mistral-saba-24b"
            ]
        )
    else:  # OpenRouter
        show_only_free = st.checkbox("Show only free models", value=True)
        if show_only_free:
            model_list = OPENROUTER_FREE_MODELS
        else:
            # List all models or you could add more paid models here
            model_list = OPENROUTER_FREE_MODELS + [
                "openai/o1-pro",
                "meta-llama/llama-3.1-70b-instruct",
                "anthropic/claude-3.7-sonnet",
                "mistralai/mistral-large-2411",
                "meta-llama/llama-3.2-90b-vision-instruct"
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
        st.markdown(f"""
        <div class="title-container">
            <span class="persona-icon">{persona['icon']}</span>
            <div class="title-text">
                <h1>{persona['title']}</h1>
                <p>{persona['subtitle']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
        with st.container():
            user_input = st.text_area(f"Message {persona['title']}...", key="user_input", height=100)
            
            col1, col2 = st.columns([5, 1])
            with col2:
                send_pressed = st.button("Send", key="send", use_container_width=True)
                
            if send_pressed and user_input:
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
                                max_tokens=4096,
                                top_p=0.95,
                                stream=True,
                                stop=None,
                            )
                            
                            # Process streaming response
                            for chunk in completion:
                                if chunk.choices[0].delta.content:
                                    response += chunk.choices[0].delta.content
                            
                            return response
                        
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
                                st.error(f"OpenRouter API Error: {response.status_code}, {response.text}")
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
                
                # Rerun to update UI (this will clear the input when the page refreshes)
                st.rerun()
    else:
        st.info("Please select a persona from the left panel to start chatting.")

# JavaScript to make cards clickable and improve UI interaction
st.markdown("""
<script>
    // This is where we would add JavaScript to enhance the UI
    // Unfortunately, Streamlit doesn't support direct JavaScript integration
    // The buttons we added serve as the clickable elements instead
</script>
""", unsafe_allow_html=True)
