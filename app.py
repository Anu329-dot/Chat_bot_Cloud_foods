import streamlit as st
import joblib
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

download_nltk_data()

# Business Information
BUSINESS_INFO = {
    "owner_name": "Aswin Krishna T and Mohan",
    "address": "Mahindra World City, Chennai",
    "phone": "+91 96294 3****",
    "hours": "6:00 PM to 10:00 PM, Monday through Sunday",
    "delivery_radius": "10-KM radius",
    "delivery_time": "30-45 minutes"
}

# Menu Configuration
MENU_CONFIG = {
    "main_courses": [
        {"name": "Chapathi + Pepper Chicken", "price": "₹145", "description": "Fluffy chapathi paired with succulent pepper-crusted chicken in a bold, spicy gravy"},
        {"name": "Idiyappam + Pepper Chicken", "price": "₹129", "description": "Soft, steamed rice noodles paired with bold pepper chicken in a perfectly balanced meal."},
        {"name": "Kuska + Potato Masala", "price": "₹145", "description": "Aromatic kuska rice paired with creamy, spiced potato masala for a satisfying meal."}
    ],
    "desserts": [
        {"name": "Brownies", "price": "₹9.99", "description": "Rich chocolate brownies"},
        {"name": "Cookies", "price": "₹5.99", "description": "Freshly baked chocolate chip cookies"}
    ]
}

def format_menu_response():
    main_courses = [f"{item['name']} - {item['price']}" for item in MENU_CONFIG['main_courses']]
    desserts = [f"{item['name']} - {item['price']}" for item in MENU_CONFIG['desserts']]
    menu = f"• Main Courses: {', '.join(main_courses)} • Desserts: {', '.join(desserts)} Would you like to place an order?"
    return menu

# Intent responses
intents = [
    {
        "tag": "greeting",
        "responses": [
            "Hello! Welcome to Cloud Foods! How can I help you today?",
            "Hi there! Ready to explore our delicious menu?",
            "Hey! Great to see you at Cloud Foods. What can I do for you?",
            "Good day! Welcome to Cloud Foods. How may I assist you?",
            "Hello! Hungry for some tasty deliciousness? I'm here to help!"
        ]
    },
    {
        "tag": "goodbye",
        "responses": [
            "Thank you for visiting Cloud Foods! We hope to see you again soon!",
            "Goodbye! Can't wait to serve you again at Cloud Foods!",
            "See you later! Your next delicious meal is just an order away!",
            "Take care! Remember, Cloud Foods is always here when you're hungry!",
            "Farewell! We'll be here ready for your next food adventure!"
        ]
    },
    {
        "tag": "thanks",
        "responses": [
            "You're welcome! Is there anything else I can help you with?",
            "Happy to help! What else can I do for you today?",
            "My pleasure! Need anything else from Cloud Foods?",
            "You're very welcome! Ready for your next request!",
            "Glad I could assist! What's next on your food journey?"
        ]
    },
    {
        "tag": "owner_info",
        "responses": [
            f"Cloud Foods is proudly owned and managed by {BUSINESS_INFO['owner_name']}.",
            f"Our wonderful restaurant is run by {BUSINESS_INFO['owner_name']}, who founded Cloud Foods.",
            f"{BUSINESS_INFO['owner_name']} are the proud owners and visionaries behind Cloud Foods!",
            f"The masterminds behind Cloud Foods are our owners {BUSINESS_INFO['owner_name']}.",
            f"Cloud Foods was established and is managed by {BUSINESS_INFO['owner_name']}."
        ]
    },
    {
        "tag": "location",
        "responses": [
            f"We are located at {BUSINESS_INFO['address']}. We deliver within a 10-km radius!",
            f"Find us at {BUSINESS_INFO['address']}! We cover a 10-km delivery area.",
            f"Our restaurant is at {BUSINESS_INFO['address']}, serving within 10-km.",
            f"Cloud Foods is situated at {BUSINESS_INFO['address']}. We serve a 10-km radius!"
        ]
    },
    {
        "tag": "contact",
        "responses": [
            f"You can reach us at {BUSINESS_INFO['phone']}. For faster service, place orders through me!",
            f"Call us at {BUSINESS_INFO['phone']}! Though I can help you order right here.",
            f"Our phone number is {BUSINESS_INFO['phone']}, but I'm here to take your order instantly!",
            f"Contact us at {BUSINESS_INFO['phone']}. Need to place an order? I can help with that!",
            f"Reach us at {BUSINESS_INFO['phone']}. For quick ordering, I'm your best bet!"
        ]
    },
    {
        "tag": "hours",
        "responses": [
            "We're open from 6:00 PM to 10:00 PM, every day of the week!",
            "Cloud Foods serves delicious meals from 6:00 PM to 10:00 PM daily.",
            "Our hours are 6:00 PM through 10:00 PM, Monday to Sunday!",
            "You can enjoy our food from 6:00 PM to 10:00 PM, all week long!"
        ]
    },
    {
        "tag": "todays_menu",
        "responses": [f"Here is today's menu at Cloud Foods: {format_menu_response()}"]
    },
    {
        "tag": "item_availability",
        "responses": [f"Here is today's menu at Cloud Foods: {format_menu_response()}"]
    },
    {
        "tag": "ingredients_inquiry",
        "responses": [
            "That dish contains fresh, high-quality ingredients and is prepared daily. It's suitable for most diets!",
            "Our dishes are made with premium ingredients. This one is quite popular and meets standard dietary requirements.",
            "We use only the finest ingredients in all our dishes. This particular item is prepared with care and quality.",
            "All our ingredients are fresh and carefully selected. This dish meets our high standards for quality and taste!",
            "We pride ourselves on using top-quality ingredients. This dish is crafted for excellent flavor and satisfaction."
        ]
    },
    {
        "tag": "place_order",
        "responses": [
            "Great! Let's get your order started. Please use this format: 'Order: Dish Name, Quantity'. Example: 'Order: Pizza, 2'",
            "Excellent! Ready to take your order. Format your order like: 'Order: Dish Name, Quantity' such as 'Order: Burger, 1'",
            "Perfect! I'll help you order. Just tell me: 'Order: Dish Name, Quantity' - for instance 'Order: Pasta, 3'",
            "Awesome! Let's place your order. Use: 'Order: Dish Name, Quantity' like 'Order: Salad, 1'",
            "Wonderful! I'm here to take your order. Please use: 'Order: Dish Name, Quantity' example: 'Order: Brownies, 4'"
        ]
    },
    {
        "tag": "order_status",
        "responses": [
            "Please provide your order ID, and I'll check the status for you right away!",
            "I can check your order status! Just share your order ID with me.",
            "Let me look up your order! What's your order ID number?",
            "I'll track your order for you! Please provide your order ID.",
            "Ready to check your order status! Just need your order ID please."
        ]
    },
    {
        "tag": "payments",
        "responses": [
            "We accept Cash on Delivery, Credit/Debit Cards, PayPal, and Google Pay!",
            "You can pay with Cash on Delivery, all major cards, or digital wallets like PayPal!",
            "Payment options include COD, credit/debit cards, and digital payments like Google Pay!",
            "We take Cash on Delivery, card payments, and digital wallets including PayPal!",
            "Choose from Cash on Delivery, card payments, or digital options like Google Pay!"
        ]
    },
    {
        "tag": "delivery_time",
        "responses": [
            "Our average delivery is 30-45 minutes, depending on location and order volume!",
            "You can expect your food in 30-45 minutes, based on your area and how busy we are!",
            "Delivery typically takes 30-45 minutes, varying with location and current orders!",
            "We aim for 30-45 minute delivery, depending on where you are and order traffic!",
            "Your food will arrive in about 30-45 minutes, adjusted for location and busy times!"
        ]
    },
    {
        "tag": "cancel_order",
        "responses": [
            "I can help cancel your order! Please provide your order ID. Note: orders being prepared may not be cancellable.",
            "Let me cancel that for you! Share your order ID. Orders in preparation might not be eligible for cancellation.",
            "I'll process your cancellation! Order ID please. Preparing orders may not be cancellable.",
            "Ready to cancel your order! Just need your order ID. Note: ongoing preparations might prevent cancellation.",
            "I can cancel your order! Provide your order ID. Orders being cooked may not be cancellable."
        ]
    },
    {
        "tag": "help",
        "responses": [
            "I can help with: today's menu, placing orders, delivery info, payment methods, and order status! What do you need?",
            "Here's what I can do: show today's menu, take orders, check delivery times, explain payments, and track orders! How can I assist?",
            "I'm here to: display our menu, process orders, provide delivery estimates, explain payment options, and check order status! What would you like?",
            "My capabilities include: menu information, order processing, delivery details, payment info, and order tracking! How can I help you?",
            "I can assist with: daily menu, food ordering, delivery timelines, payment methods, and order updates! What do you need help with?"
        ]
    },
    {
        "tag": "fallback",
        "responses": [
            "I'm sorry, I didn't quite understand that. I'm here to help with Cloud Foods' menu, orders, and information. Could you please rephrase your question?",
            "I'm not sure I follow. I specialize in Cloud Foods services like menu info and ordering. Could you try asking differently?",
            "That's beyond my food expertise! Try asking about our menu, how to order, or our restaurant information.",
            "I didn't catch that. Remember, I'm your Cloud Foods assistant! Ask me about food orders, menu items, or delivery.",
            "Let's stick to food topics! I can help with menu questions, placing orders, or restaurant info. What would you like to know?"
        ]
    }
]

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'best_model.pkl', 'vectorizer.pkl', and 'label_encoder.pkl' are in the same directory.")
        return None, None, None

model, vectorizer, label_encoder = load_models()

# Chatbot response function
def chatbot_response(user_input):
    if model is None or vectorizer is None or label_encoder is None:
        return "I'm sorry, the chatbot is not properly configured. Please check the model files.", "error"
    
    input_text = vectorizer.transform([user_input])
    predicted_intent = model.predict(input_text)[0]
    prediction = label_encoder.inverse_transform([predicted_intent])[0]
    
    for intent in intents:
        if intent['tag'] == prediction:
            response = random.choice(intent['responses'])
            break
    else:
        response = "I'm sorry, I didn't quite understand that. Could you please rephrase?"
    
    return response, prediction

# Page config
st.set_page_config(page_title="Cloud Foods", page_icon="☁️", layout="wide")

# Custom CSS for enhanced yellow and white theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #FFF9E6 0%, #FFFEF7 50%, #FFF5D6 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FFD700 100%);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(255, 165, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .main-header h1 {
        color: #1a1a1a;
        margin: 0;
        font-size: 3em;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1;
    }
    
    .main-header .subtitle {
        color: #333;
        margin: 10px 0 0 0;
        font-size: 1.3em;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .bot-name {
        background: linear-gradient(90deg, #FFD700, #FFA500, #FFD700);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-flow 3s ease infinite;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    @keyframes gradient-flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .chat-message {
        padding: 18px 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        display: flex;
        align-items: flex-start;
        animation: slideIn 0.4s ease-out;
        max-width: 75%;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%);
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    .bot-message {
        background: white;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 2px solid #FFD700;
    }
    
    .message-icon {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        margin-right: 12px;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .user-icon {
        background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%);
        color: white;
    }
    
    .bot-icon {
        background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%);
        color: #1a1a1a;
    }
    
    .message-content {
        flex: 1;
        color: #1a1a1a;
        line-height: 1.6;
        font-size: 1em;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        border: none;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
        box-shadow: 0 4px 10px rgba(255, 165, 0, 0.3);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255, 165, 0, 0.4);
        border: 2px solid #FF8C00;
        background: linear-gradient(135deg, #FFC107 0%, #FF8C00 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .stChatInput>div>div>textarea {
        border: 3px solid #FFD700 !important;
        border-radius: 25px !important;
        padding: 15px 20px !important;
        font-size: 1em !important;
        background: white !important;
        box-shadow: 0 2px 10px rgba(255, 215, 0, 0.2) !important;
    }
    
    .stChatInput>div>div>textarea:focus {
        border-color: #FFA500 !important;
        box-shadow: 0 4px 20px rgba(255, 165, 0, 0.3) !important;
    }
    
    .quick-actions {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
        margin: 20px 0;
        padding: 15px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stHorizontalBlock"] {
        gap: 10px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>☁️ Cloud Foods</h1>
        <p> Your AI Food Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Welcome to Cloud Foods! I'm Cloud Foods, your friendly AI assistant. How can I help you today? 😊"}
    ]

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-icon user-icon">👤</div>
                <div class="message-content">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-icon bot-icon">☁️</div>
                <div class="message-content">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response, intent = chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Quick action buttons
st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📋 Menu"):
        st.session_state.messages.append({"role": "user", "content": "Show me the menu"})
        response, intent = chatbot_response("Show me the menu")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col2:
    if st.button("📍 Location"):
        st.session_state.messages.append({"role": "user", "content": "Where are you located?"})
        response, intent = chatbot_response("Where are you located?")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col3:
    if st.button("🕒 Hours"):
        st.session_state.messages.append({"role": "user", "content": "What are your hours?"})
        response, intent = chatbot_response("What are your hours?")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col4:
    if st.button("❓ Help"):
        st.session_state.messages.append({"role": "user", "content": "Help"})
        response, intent = chatbot_response("Help")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col5:
    if st.button("🔄 Clear"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Welcome to Cloud Foods! I'm Cloud Foods, your friendly AI assistant. How can I help you today? 😊"}
        ]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)