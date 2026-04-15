import re
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras.applications.efficientnet import preprocess_input
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import fitz  # PyMuPDF
import random
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from numpy.typing import NDArray
from typing import Dict

###############################################################################################

# STUN server config for WebRTC (add TURN for production)
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)


# --- Video Frame Processor ---
class CameraProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

##################3


# ✅ Page Configuration
st.set_page_config(page_title="🌿 Plant Assistant", layout="wide")
###############
custom_css = """
<style>
    /* --- NEW COMPREHENSIVE CSS TO REDUCE TOP SPACE --- */

    /* Reset default browser margins for html and body */
    html, body {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Target the main Streamlit app container */
    .stApp {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }

   

    
    /* Another common block that might add vertical space (e.g., for vertical stacking of elements) */
    div[data-testid="stVerticalBlock"] {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }


   

    /* --- START OF YOUR ORIGINAL CUSTOM CSS (keep all your current styles below this line) --- */

    /* Global Font Change (from your existing code) */
    html, body, [class*="st-emotion-cache"], .stApp {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        color: #333333; /* Darker default text color */
    }

    /* Sidebar Styling (from your existing code) */
    [data-testid="stSidebar"] {
        background-color: #e0ecba; /* Your chosen hex color, replace #e0ffe0 if you haven't */
        padding-top: 20px;
    }
    [data-testid="stSidebarContent"] {
        background-color: #e0ecba; /* Your chosen hex color, replace #e0ffe0 if you haven't */
    }

    /* Navigation (st.radio) in Sidebar (from your existing code) */
    div.stRadio > label {
        color: #1a7a37;
        font-weight: bold;
        padding: 8px 15px;
        margin: 5px 0;
        border-radius: 8px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    div.stRadio > label:hover {
        background-color: #d4edda;
        color: #1a7a37;
    }
   
   
    
    
   
   
</style>
"""

# Inject the CSS at the top of the app
st.markdown(custom_css, unsafe_allow_html=True)



###############

# ✅ Load Plant Data
def load_plant_data(directory):
    image_paths = []
    image_labels = []

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(label_path, filename))
                    image_labels.append(label)

    return image_paths, image_labels

image_paths, image_labels = load_plant_data(r"C:\Users\sujic\Downloads\Plant Model\house_plant_species")
data = pd.DataFrame({
    "image": image_paths,
    "name": image_labels
})

# ✅ Load Trained Model and Class Names
model = tf.keras.models.load_model("improved_plant_classifier_model.keras")
class_names = [
    'African Violet (Saintpaulia ionantha)', 'Aloe Vera', 'Anthurium (Anthurium andraeanum)',
    'Areca Palm (Dypsis lutescens)', 'Begonia (Begonia spp.)', 'Bird of Paradise (Strelitzia reginae)',
    'Calathea', 'Christmas Cactus (Schlumbergera bridgesii)', 'Chrysanthemum',
    'Daffodils (Narcissus spp.)', 'Dumb Cane (Dieffenbachia spp.)', 'Hyacinth (Hyacinthus orientalis)',
    'Jade plant (Crassula ovata)', 'Kalanchoe', 'Lilium (Hemerocallis)',
    'Lily of the valley (Convallaria majalis)', 'Monstera Deliciosa (Monstera deliciosa)', 'Orchid',
    'Peace lily', 'Poinsettia (Euphorbia pulcherrima)', 'Polka Dot Plant (Hypoestes phyllostachya)',
    'Rubber Plant (Ficus elastica)', 'Snake plant (Sanseviera)', 'Tradescantia', 'Tulip'
]

# ✅ Load care tips from PDF
def load_care_tips_from_pdf(pdf_path):
    care_tips = {}
    doc = fitz.open(pdf_path)
    text = "\n".join([doc.load_page(i).get_text("text") for i in range(doc.page_count)])
    for plant in class_names:
        try:
            start = text.index(plant)
            end = min([text.find(p, start + 1) for p in class_names if p != plant and text.find(p, start + 1) != -1] + [len(text)])
            care_section = text[start:end]
            plant_care = {
                "Watering": re.search(r"Watering: (.*?)(?=\n|$)", care_section, re.S),
                "Space": re.search(r"Space: (.*?)(?=\n|$)", care_section, re.S),
                "Benefits": re.search(r"Benefits: (.*?)(?=\n|$)", care_section, re.S),
                "Allergy/Toxicity": re.search(r"Allergy/Toxicity: (.*?)(?=\n|$)", care_section, re.S),
                "Placement": re.search(r"Placement: (.*?)(?=\n|$)", care_section, re.S),
                "Care Tips": re.search(r"Care Tips: (.*?)(?=\n|$)", care_section, re.S),
                "Fun Fact": re.search(r"Fun Fact: (.*?)(?=\n|$)", care_section, re.S)
            }
            care_tips[plant] = {k: v.group(1).strip() if v else "N/A" for k, v in plant_care.items()}
        except Exception:
            care_tips[plant] = {k: "N/A" for k in ["Watering", "Space", "Benefits", "Allergy/Toxicity", "Placement", "Care Tips", "Fun Fact"]}
    return care_tips

care_tips = load_care_tips_from_pdf("Houseplant_Guide_Extended.pdf")

# ✅ UI Helper Functions
def display_care_tips(plant_name, tips):
    st.markdown(f"""
        <div style="background-color:#f0fff4; padding: 20px; border-radius: 15px; border: 2px solid #34d399;">
            <h2 style="color:#047857;">🌱 Care Tips for <i>{plant_name}</i></h2>
            <ul style="list-style: none; padding-left: 0;">
                {''.join([f'<li><strong style="color:#10b981;">{k}:</strong> {v}</li>' for k,v in tips.items()])}
            </ul>
        </div>
    """, unsafe_allow_html=True)

def styled_prediction(name, conf):
    st.markdown(f"""
        <div style='background-color:#ecfdf5; padding:10px; border-radius:10px; border-left:5px solid #34d399;'>
            <strong style='color:#065f46;'>{name}</strong> — <span style='color:#059669;'>{conf:.2f}% confidence</span>
        </div>
    """, unsafe_allow_html=True)

# ✅ Prediction Function with Timing
def process_and_predict(image_file):
    start_time = time.time()
    img = image.load_img(image_file, target_size=(260, 260))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)[0]
    end_time = time.time()
    pred_time = end_time - start_time
    top_indices = prediction.argsort()[-3:][::-1]
    top_predictions = [(class_names[i], float(prediction[i]) * 100) for i in top_indices]
    img_display = Image.open(image_file).convert("RGB")
    return top_predictions, img_display, pred_time


# ✅ Define tabs with emojis for better UI
tabs = {
    "🌿 Identify Plant": "Identify Plant",
    "💬 Chatbot": "Chatbot",
    "🧠 Quiz": "Quiz",
    "🌱 Guess the Plant": "Guess the Plant",
    "ℹ️ About Us": "About Us"
}

# Sidebar with styled header
st.sidebar.markdown("## Navigation")
selected_label = st.sidebar.radio("Choose a tab:", list(tabs.keys()))

# Get the internal tab name
selected_tab = tabs[selected_label]


###########################333

# ✅ Tab 1: Identify Plant
if selected_tab == "Identify Plant":
    st.header("🌿 Plant Identifier & Care Assistant")

    # Add a unique key argument to the radio button for input method
    input_method = st.radio("Choose input method:", ["Upload Image", "Use Camera"], key="input_method_radio")

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"], key="upload_image")
        if uploaded_file:
            top_preds, img, pred_time = process_and_predict(uploaded_file)
            st.image(img, caption="Uploaded Image", width=200)

            st.markdown("### 🌿 Top 3 Prediction Probabilities:")
            for i, (name, conf) in enumerate(top_preds):
                # ... (rest of your image prediction display code)
                top_prediction_plant = top_preds[0][0]
                st.success(f"✅ Predicted as: **{top_prediction_plant}**")
                st.success(f"⏱️ Prediction completed in {pred_time:.2f} seconds.")
                
                if top_prediction_plant in care_tips:
                    display_care_tips(top_prediction_plant, care_tips[top_prediction_plant])

    elif input_method == "Use Camera":
        # Use st.camera_input() to get an image from the camera
        captured_image = st.camera_input("Take a picture of your plant", key="camera_input") # Added a key
        if captured_image:
            try:
                # captured_image is a BytesIO object, so we need to open it as a PIL Image
                image = Image.open(captured_image)
                st.image(image, caption="Captured Image", use_container_width=True)
                st.markdown("---")

                # --- Rest of your camera processing and prediction code ---
                img_array = image.resize((260, 260))
                img_array = np.array(img_array) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0]
                top_indices = prediction.argsort()[-3:][::-1]
                top_preds = [(class_names[i], float(prediction[i]) * 100) for i in top_indices]

                st.markdown("### 🌿 Top 3 Prediction Probabilities:")
                for i, (name, conf) in enumerate(top_preds):
                    if i == 0:
                        st.markdown(f"""
                            <div style='background-color:#d1fae5; padding:15px; border-radius:12px; border-left:6px solid #059669;'>
                                <strong style='color:#065f46; font-size:20px;'>🌟 {name}</strong><br>
                                <span style='color:#047857; font-size:16px;'>Confidence: {conf:.2f}%</span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style='background-color:#f0fdf4; padding:10px; border-radius:10px; border-left:4px solid #a7f3d0;'>
                                <strong style='color:#065f46;'>{name}</strong><br>
                                <span style='color:#059669;'>{conf:.2f}% confidence</span>
                            </div>
                        """, unsafe_allow_html=True)

                top_prediction_plant = top_preds[0][0]
                st.success(f"✅ Predicted as: **{top_prediction_plant}**")
                # Prediction time is not directly available here, you'd need to time the prediction
                if top_prediction_plant in care_tips:
                    display_care_tips(top_prediction_plant, care_tips[top_prediction_plant])

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Tab 2: Chatbot ---
elif selected_tab == "Chatbot":
    st.header("🤖 Plant Care Chatbot Assistant")
    st.markdown("Ask me about any houseplant care. For example: _'How do I water Aloe Vera?'_")

    # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Define houseplant data (ensure it's defined here or imported globally)
    houseplants = {
    "african violet": {
        "watering": "Keep soil moist but not soggy.",
        "space": "Compact, 6-12 inches wide.",
        "benefits": "Beautiful blooms; aesthetic value.",
        "toxicity": "Safe for pets; suitable for pollen allergy sufferers.",
        "placement": "Bright, indirect light; ideal for windowsills in living rooms or bedrooms.",
        "care": "Rotate for even growth; fertilize every 2 weeks.",
        "fun": "Can bloom year-round!"
    },
    "aloe vera": {
        "watering": "Allow soil to dry completely.",
        "space": "12-24 inches tall.",
        "benefits": "Medicinal; air purifier.",
        "toxicity": "Toxic to pets; not recommended for latex allergy sufferers.",
        "placement": "Bright, direct or indirect sunlight; best in kitchens or sunny bathroom windowsills.",
        "care": "Use cactus soil; prune pups.",
        "fun": "Contains over 75 active compounds."
    },
    "anthurium": {
        "watering": "Moist soil, good drainage.",
        "space": "12-18 inches.",
        "benefits": "Decorative; air purifier.",
        "toxicity": "Toxic; not suitable for sensitive skin or pollen allergies.",
        "placement": "Bright, indirect light; excellent for living rooms or offices with stable humidity.",
        "care": "Wipe leaves; repot biennially.",
        "fun": "Known as Flamingo Flower."
    },
    "areca palm": {
        "watering": "Keep soil moist in summer.",
        "space": "Up to 6-7 feet.",
        "benefits": "Humidifier; air purifier.",
        "toxicity": "Non-toxic; very allergy-friendly.",
        "placement": "Bright, indirect light; perfect for corners of living rooms or hallways.",
        "care": "Mist regularly.",
        "fun": "Called Butterfly Palm."
    },
    "begonia": {
        "watering": "When top soil is dry.",
        "space": "6 inches to 2 feet.",
        "benefits": "Flowers; aesthetic.",
        "toxicity": "Toxic to pets; pollen minimal.",
        "placement": "Indirect light; great on shelves or tables in well-lit rooms.",
        "care": "Deadhead blooms.",
        "fun": "Over 1,800 species worldwide."
    },
    "bird of paradise": {
        "watering": "Water when top inch is dry.",
        "space": "Up to 6 feet.",
        "benefits": "Tropical aesthetic; air purification.",
        "toxicity": "Toxic to pets.",
        "placement": "Bright light; large space areas like living rooms.",
        "care": "Keep leaves clean.",
        "fun": "Resembles a bird in flight."
    },
    "calathea": {
        "watering": "Keep soil evenly moist.",
        "space": "1-2 feet.",
        "benefits": "Decorative leaves.",
        "toxicity": "Non-toxic; pet friendly.",
        "placement": "Low to medium indirect light; bedrooms or bathrooms.",
        "care": "High humidity needed.",
        "fun": "Leaves close at night."
    },
    "christmas cactus": {
        "watering": "Water when dry to touch.",
        "space": "6-12 inches.",
        "benefits": "Flowers in winter.",
        "toxicity": "Non-toxic.",
        "placement": "Indirect light; cool rooms.",
        "care": "Let rest after bloom.",
        "fun": "Can live for decades."
    },
    "chrysanthemum": {
        "watering": "Keep soil moist.",
        "space": "12-36 inches.",
        "benefits": "Bright flowers; air purification.",
        "toxicity": "Toxic to pets.",
        "placement": "Bright indirect light; seasonal indoor use.",
        "care": "Pinch stems for bushiness.",
        "fun": "Symbol of joy in many cultures."
    },
    "daffodils": {
        "watering": "Moderate.",
        "space": "6-18 inches.",
        "benefits": "Early bloomers; cheerfulness.",
        "toxicity": "Toxic to pets and humans.",
        "placement": "Bright indoor pots.",
        "care": "Let foliage die back.",
        "fun": "Associated with spring."
    },
    "dumb cane": {
        "watering": "Keep soil moist.",
        "space": "3-6 feet.",
        "benefits": "Large foliage.",
        "toxicity": "Toxic; irritates skin and mouth.",
        "placement": "Indirect light; office corners.",
        "care": "Do not let dry out.",
        "fun": "Nicknamed for speech effects if ingested."
    },
    "hyacinth": {
        "watering": "Water moderately.",
        "space": "6-12 inches.",
        "benefits": "Fragrant flowers.",
        "toxicity": "Toxic to pets.",
        "placement": "Bright spot; seasonal.",
        "care": "Cool temps help blooms last.",
        "fun": "Very fragrant spring bulb."
    },
    "jade plant": {
        "watering": "Let soil dry between waterings.",
        "space": "1-3 feet.",
        "benefits": "Symbol of prosperity.",
        "toxicity": "Mildly toxic to pets.",
        "placement": "Bright light; ideal for desks.",
        "care": "Avoid overwatering.",
        "fun": "Also called money plant."
    },
    "kalanchoe": {
        "watering": "Let soil dry between watering.",
        "space": "6 inches to 1.5 feet.",
        "benefits": "Colorful blooms.",
        "toxicity": "Toxic to pets.",
        "placement": "Bright indirect light.",
        "care": "Deadhead for more blooms.",
        "fun": "Succulent flowering plant."
    },
    "lilium": {
        "watering": "Moderate; avoid soggy soil.",
        "space": "1-4 feet.",
        "benefits": "Showy flowers.",
        "toxicity": "Very toxic to cats.",
        "placement": "Bright light; only indoors seasonally.",
        "care": "Stake tall stems.",
        "fun": "Daylily varieties are edible."
    },
    "lily of the valley": {
        "watering": "Moist soil.",
        "space": "6-12 inches.",
        "benefits": "Fragrant.",
        "toxicity": "Highly toxic.",
        "placement": "Cool indirect light.",
        "care": "Divide when crowded.",
        "fun": "National flower of Finland."
    },
    "monstera deliciosa": {
        "watering": "When top soil is dry.",
        "space": "Up to 10 feet.",
        "benefits": "Stylish; air purifier.",
        "toxicity": "Toxic if ingested.",
        "placement": "Bright indirect light; large indoor spaces.",
        "care": "Use moss poles.",
        "fun": "Leaves develop natural holes."
    },
    "orchid": {
        "watering": "Weekly.",
        "space": "1-3 feet.",
        "benefits": "Elegant flowers.",
        "toxicity": "Non-toxic.",
        "placement": "Bright, indirect light.",
        "care": "Use orchid potting mix.",
        "fun": "Symbol of beauty."
    },
    "peace lily": {
        "watering": "Keep soil moist.",
        "space": "1-4 feet.",
        "benefits": "Air purification; increases humidity.",
        "toxicity": "Toxic to pets.",
        "placement": "Indirect light; great for bathrooms or bedrooms.",
        "care": "Prune yellow leaves.",
        "fun": "Blooms even in low light."
    },
    "poinsettia": {
        "watering": "Moderate.",
        "space": "1-3 feet.",
        "benefits": "Festive decor.",
        "toxicity": "Mildly toxic.",
        "placement": "Bright filtered light.",
        "care": "Do not expose to cold drafts.",
        "fun": "Iconic holiday plant."
    },
    "polka dot plant": {
        "watering": "Keep moist.",
        "space": "6-12 inches.",
        "benefits": "Colorful foliage.",
        "toxicity": "Mildly toxic.",
        "placement": "Indirect light; great for tabletops.",
        "care": "Pinch for bushiness.",
        "fun": "Named for its spotted leaves."
    },
    "rubber plant": {
        "watering": "Let soil dry out slightly.",
        "space": "3-10 feet.",
        "benefits": "Removes toxins; large leaves.",
        "toxicity": "May irritate skin.",
        "placement": "Indirect bright light.",
        "care": "Clean leaves monthly.",
        "fun": "Latex-producing tree."
    },
    "snake plant": {
        "watering": "Minimal; drought-tolerant.",
        "space": "1-4 feet.",
        "benefits": "Air purification.",
        "toxicity": "Mildly toxic.",
        "placement": "Any light level.",
        "care": "Do not overwater.",
        "fun": "Converts CO2 to oxygen at night."
    },
    "tradescantia": {
        "watering": "Water when top soil is dry.",
        "space": "6-18 inches.",
        "benefits": "Colorful trailing plant.",
        "toxicity": "Non-toxic.",
        "placement": "Hanging baskets in light rooms.",
        "care": "Prune for fullness.",
        "fun": "Also called wandering dude."
    },
    "tulip": {
        "watering": "Keep soil moist.",
        "space": "10-20 inches.",
        "benefits": "Colorful spring flowers.",
        "toxicity": "Toxic to pets.",
        "placement": "Cool, bright indoor spots.",
        "care": "Grow from bulbs.",
        "fun": "National flower of the Netherlands."
    }
}


    def normalize(text):
        return text.lower().strip()

    def answer_question(user_input):
        user_input = normalize(user_input)

        for plant in houseplants:
            if plant in user_input:
                info = houseplants[plant]
                if "water" in user_input:
                    return f"💧 **{plant.title()} Watering**: {info['watering']}"
                elif "space" in user_input or "size" in user_input or "height" in user_input:
                    return f"📏 **{plant.title()} Size**: {info['space']}"
                elif "benefit" in user_input:
                    return f"🌟 **{plant.title()} Benefits**: {info['benefits']}"
                elif "toxic" in user_input or "allergy" in user_input or "pet" in user_input:
                    return f"⚠️ **{plant.title()} Toxicity/Allergy**: {info['toxicity']}"
                elif "place" in user_input or "location" in user_input or "where" in user_input:
                    return f"🏡 **{plant.title()} Placement**: {info['placement']}"
                elif "care" in user_input or "tip" in user_input:
                    return f"🛠️ **{plant.title()} Care Tips**: {info['care']}"
                elif "fun" in user_input or "fact" in user_input:
                    return f"🎉 **{plant.title()} Fun Fact**: {info['fun']}"
                else:
                    return f"🤔 What would you like to know about **{plant.title()}**? You can ask about watering, placement, toxicity, benefits, care, or fun facts."
        return "❌ Sorry, I couldn’t identify the plant. Please use the full or common name (e.g., Aloe Vera, Snake Plant)."

    # Create session state for conversation
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("💬 Ask me about plant care:")

    if user_input:
        response = answer_question(user_input)
        st.session_state.conversation.append(("You", user_input))
        st.session_state.conversation.append(("Bot", response))

    st.markdown("---")
    for speaker, message in st.session_state.conversation:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation = []



# --- Tab 3: Quiz ---
elif selected_tab == "Quiz":
    import random

    st.header("🌱 The Ultimate Plant Quiz")

    # Full quiz question set (extended)
    full_quiz_data = [
        {
            "header": "🌿 Challenge 1",
            "question": "Which plant famously purifies your air while you sleep?",
            "options": ["Peace Lily", "Snake Plant", "Orchid", "Tulip"],
            "answer": "Snake Plant"
        },
        {
            "header": "🌿 Challenge 2",
            "question": "Which of these plants defies winter, blooming when it's cold outside?",
            "options": ["Christmas Cactus", "Calathea", "Aloe Vera", "Bird of Paradise"],
            "answer": "Christmas Cactus"
        },
        {
            "header": "🌿 Challenge 3",
            "question": "Among these, which plant is considered safe for your beloved pets?",
            "options": ["African Violet", "Aloe Vera", "Lily of the Valley", "Dumb Cane"],
            "answer": "African Violet"
        },
        {
            "header": "🌿 Challenge 4",
            "question": "Which plant is known for its water-storing leaves?",
            "options": ["Aloe Vera", "Fiddle Leaf Fig", "Monstera", "Bamboo Palm"],
            "answer": "Aloe Vera"
        },
        {
            "header": "🌿 Challenge 5",
            "question": "If you forget to water it, which indoor plant is most likely to survive?",
            "options": ["Snake Plant", "Fern", "Peace Lily", "Areca Palm"],
            "answer": "Snake Plant"
        },
        {
            "header": "🌿 Challenge 6",
            "question": "Which plant is often used in herbal tea for relaxation?",
            "options": ["Lavender", "Mint", "Chamomile", "Basil"],
            "answer": "Chamomile"
        }
    ]

    # Initialize session state
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_reset" not in st.session_state:
        st.session_state.quiz_reset = True

    # Shuffle and select 3 random questions
    if st.session_state.quiz_reset:
        st.session_state.shuffled_quiz_data = random.sample(full_quiz_data, 3)
        st.session_state.user_answers = {}
        st.session_state.quiz_reset = False

    # Quiz Form (before submission)
    if not st.session_state.quiz_submitted:
        for i, q in enumerate(st.session_state.shuffled_quiz_data):
            st.subheader(q["header"])
            st.markdown(f"**{q['question']}**")
            key = f"quiz_{i}"
            selected = st.radio(" ", q["options"], key=key)
            st.session_state.user_answers[q["question"]] = selected

        if st.button("Submit Quiz"):
            if all(ans is not None for ans in st.session_state.user_answers.values()):
                st.session_state.quiz_submitted = True
                st.rerun()
            else:
                st.warning("Please answer all the questions before submitting.")

    # Quiz Results
    if st.session_state.quiz_submitted:
        score = 0
        st.subheader("🎉 Quiz Results")
        for q in st.session_state.shuffled_quiz_data:
            user_answer = st.session_state.user_answers.get(q["question"])
            correct = q["answer"]
            st.subheader(q["header"])
            st.markdown(f"**{q['question']}**")
            if user_answer == correct:
                st.success(f"✅ {user_answer} (Correct!)")
                score += 1
            else:
                st.error(f"❌ {user_answer} (Incorrect)")
                st.info(f"✅ Correct Answer: {correct}")

        st.markdown("---")
        st.markdown(f"## 🌟 Your Score: **{score} / {len(st.session_state.shuffled_quiz_data)}**")

        # Score-based feedback
        if score == len(st.session_state.shuffled_quiz_data):
            st.balloons()
            st.success(f"**Botanical Genius!** You absolutely crushed it! Every answer was spot on!🌟")
        elif score >= len(st.session_state.shuffled_quiz_data) * 0.6:
            st.info(f"**Impressive Green Thumb!** You've got a real knack for plants! Keep blossoming!🌱")
        else:
            st.warning(f"**Budding Botanist!** Every expert started somewhere. Keep exploring the wonderful world of plants!🌿")

        # Try Again button
        if st.button("Try Again"):
            for key in list(st.session_state.keys()):
                if key.startswith("quiz_"):
                    del st.session_state[key]
            st.session_state.quiz_submitted = False
            st.session_state.quiz_reset = True
            st.rerun()


# --- Tab 4: Guess the Plant Game ---
elif selected_tab == "Guess the Plant":
    st.header("🎮 Guess the Plant Game")

    if 'data' not in globals():
        st.error("❌ 'data' variable not found. Make sure the DataFrame with 'image' and 'name' columns is defined.")
    elif data.empty:
        st.warning("⚠️ No plant images available.")
    else:
        if "game_data" not in st.session_state or st.button("🔄 New Game"):
            sampled = data.sample(3).reset_index(drop=True)
            game_entries = []

            for _, row in sampled.iterrows():
                correct_name = row["name"]
                wrong_choices = random.sample(
                    [n for n in data["name"].unique() if n != correct_name], 2
                )
                options = wrong_choices + [correct_name]
                random.shuffle(options)

                game_entries.append({
                    "image": row["image"],
                    "correct": correct_name,
                    "options": options,
                    "selected": None
                })

            st.session_state.game_data = game_entries
            st.session_state.game_submitted = False

        game_data = st.session_state.game_data

        st.markdown("### 🪴 Select the correct plant name for each image:")

        for idx, item in enumerate(game_data):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(item["image"], width=150, caption=f"Image {idx + 1}")
            with col2:
                st.markdown(
                    f"<h4 style='font-size:20px;'>🌿 Which plant is this?</h4>",
                    unsafe_allow_html=True
                )
                selected = st.radio(
                    "",
                    item["options"],
                    key=f"game_radio_{idx}",
                    index=item["options"].index(item["selected"]) if item["selected"] else 0,
                )
                st.session_state.game_data[idx]["selected"] = selected

        if st.button("✅ Submit Answers") and not st.session_state.game_submitted:
            st.session_state.game_submitted = True
            score = 0
            st.markdown("### 📝 Results:")

            for idx, item in enumerate(st.session_state.game_data):
                if item["selected"] == item["correct"]:
                    st.success(f"✅ Image {idx+1}: Correct! It is **{item['correct']}**.")
                    score += 1
                else:
                    st.error(f"❌ Image {idx+1}: You chose **{item['selected']}**, but the correct answer is **{item['correct']}**.")

            st.markdown(f"### 🎯 Your Score: **{score} / 3**")

            if score == 3:
                st.balloons()
                st.success("🌟**You nailed it** – every answer was spot on!")
            elif score == 2:
                st.info("🌼 **You really know your plants!** Keep thriving!")
            else:
                st.warning("🌱**Keep trying!** It's the beginning of a fascinating journey of discovering the plant kingdom!")

        elif st.session_state.game_submitted:
            st.info("ℹ️ You've already submitted. Click '🔄 New Game' to try again.")


# ✅ Tab 5: About Us
if selected_tab == "About Us":
    with st.container():
        st.header("🌿 About Plant Assistant")
        st.markdown("""
        **Plant Assistant** helps you with the following:

        🌱 **Identify houseplants** from photos or camera.<br>
        🧠 **Test your knowledge** with a fun plant quiz.<br>
        🌍 **Guess the Plant** feature challenges you to recognize plants based on given clues.<br>
        💬 **Chatbot** to help you interact and get information about plants in a conversational way.<br>
        
        ### Features:
        
        - **Identify Plant**: Upload a photo of a plant or use your camera to identify it.<br>
        - **Chatbot**: Ask questions and get detailed information about plants.<br>
        - **Quiz**: Test your plant knowledge with a fun and interactive quiz.<br>
        - **Guess the Plant**: A game where you are given clues, and you need to guess the plant name.<br>

        **Technologies Used:**

        - **TensorFlow + EfficientNet** for plant classification.<br>
        - **Streamlit + WebRTC** for UI and camera interaction.<br>
        
        Created with ❤️ for plant lovers.<br><br>

        **Developed by**: Madhuri & Sujata
        """, unsafe_allow_html=True)
