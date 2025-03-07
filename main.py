import streamlit as st
import random
import os
import requests
import streamlit.components.v1 as components
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

st.set_page_config(page_title="SynapseAI", page_icon="generated-icon.png", layout="wide")

API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        return None
    except KeyError:
        st.error("Error: Unexpected response format from the API.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Make sure style.css exists

analyzer = SentimentIntensityAnalyzer()

challenges = {
    "Compliment a classmate.": "😊",
    "Offer to help a friend with homework.": "🤝",
    "Write a kind note to family.": "💌",
    "Hold the door open for someone.": "🚪",
    "Smile at a stranger.": "😀",
    "Help a neighbor with yard work.": "🏡",
    "Leave an anonymous encouraging message.": "📝",
    "Pick up litter.": "🗑️",
    "Thank school staff.": "🙏",
    "Offer a seat to someone.": "💺",
    "Make someone laugh.": "😂",
    "Send a positive text.": "📱",
    "Help a sibling with a task.": "🙌",
    "Share treats with your class.": "🍪",
    "Write a positive review for a local business.": "⭐",
    "Donate clothes or toys.": "🎁",
    "Leave a kind comment online.": "💬",
    "Help a friend study.": "📚",
    "Be patient with someone having a bad day.": "😌",
    "Forgive someone.": "❤️",
    "Plant a sapling.": "🌱",
    "Visit a nursing home.": "👵👴",
    "Volunteer at a charity.": "🙋",
    "Help a pet owner.": "🐾",
    "Help a teacher after class.": "🍎",
    "Be respectful to everyone.": " Respect",
    "Listen attentively.": "👂",
    "Practice gratitude.": "🙏",
    "Be kind to yourself.": "🧘",
    "Practice self-care.": "🛀",
    "Take a break from social media.": "📵",
    "Spend time in nature.": "🌳",
    "Do something that makes you happy.": "😄",
    "Get a good night's sleep.": "😴",
    "Eat a healthy meal.": "🥗",
    "Exercise.": "🏃",
    "Drink plenty of water.": "💧",
    "Take care of your mental health.": "🧠",
}

st.image("banner.png", width=541)
st.title("SynapseAI: *Wellbeing for Students, by Students.*")
st.image("kindness.png", caption="Spreading Kindness 💖", width=100)
st.write("In today's fast-paced world, students face numerous challenges to their wellbeing, including work overload, social tensions, and the pressures of academic life. That's why we developed SynapseAI, a platform designed specifically for students. Here you'll find AI-powered conflict resolution support, journaling prompts for self-reflection, mindfulness tips and kindness challenges to help you find your calm and peace of mind in your everyday life.")
if "completed_tasks" not in st.session_state:
    st.session_state.completed_tasks = {}

if st.button("Get My Challenge! 🎁"):
    chosen_challenge = random.choice(list(challenges.keys()))
    st.write(f"{challenges[chosen_challenge]} {chosen_challenge}")
    if chosen_challenge not in st.session_state.completed_tasks:
        st.session_state.completed_tasks[chosen_challenge] = False
    checkbox_key = f"checkbox_{chosen_challenge}"

    def update_completion(chosen_challenge=chosen_challenge):
        st.session_state.completed_tasks[chosen_challenge] = st.session_state[checkbox_key]

    completed = st.checkbox("I completed this task! ✅", value=st.session_state.completed_tasks[chosen_challenge],
                            key=checkbox_key, on_change=update_completion)
st.subheader("Completed Tasks:")
for task, completed in st.session_state.completed_tasks.items():
        if completed:
            st.write(f"- {task} {challenges[task]}")

st.write("----------------------------------------------")

st.write("Share Your Kindness Story 📖: *Tell us about a time when you showed kindness, received kindness, or witnessed kindness in action.  Your story can inspire others, spread positivity, and remind us of the power of compassion.  Submit your story and let's create a ripple effect of kindness together!*")
experience = st.text_area("")

if st.button("Analyze My Story 🔍"):
    if experience:
        try:
            scores = analyzer.polarity_scores(experience)
            compound = scores['compound']
            pos = scores['pos']
            neg = scores['neg']
            neu = scores['neu']
            emotion = "neutral"
            tips = []
            if compound >= 0.5:
                emotion = "joyful"
                tips = ["Celebrate your positive experience! 🎉", "Consider sharing your story with others. 😊"]
            elif compound <= -0.5:
                emotion = "distressed"
                tips = ["Allow yourself to feel your emotions. 😔", "Reach out to someone you trust for support. 📞"]
            elif neg > 0.3 and neu > 0.5:
                emotion = "anxious"
                tips = ["Practice deep breathing to calm your nerves. 🌬️",
                        "Try a grounding technique to bring yourself to the present. 🦶"]
            elif neu > 0.7 and pos < 0.2 and neg < 0.2:
                emotion = "unchallenged"
                tips = ["Find a new activity to stimulate your mind. 🧩", "Engage in a creative outlet to express yourself. 🎨"]
            else:
                emotion = "contemplative"
                tips = ["Take a moment to reflect on your experience. 🤔",
                        "Consider how you can apply this experience to your life. 🌱"]
            st.write(f"Based on your story, you seem to be feeling {emotion}. 😌")
            for tip in tips:
                st.write(f"- {tip}")

            sentiment_data = {
                "Positive": pos,
                "Neutral": neu,
                "Negative": neg,
            }

            st.subheader("Sentiment Analysis Breakdown 📊")
            st.bar_chart(sentiment_data)

        except Exception as e:
            st.error(f"Unexpected Error: {e}")
            st.write(f"Error details: {e}")
    else:
        st.write("Please share your experience to analyze its sentiment. 📖")

def mindfulness_section():
    st.subheader("Mindfulness Exercises 🧘")
    st.write("Need a break from the studies and a moment of calm? Discover simple yet effective mindfulness techniques to bring peace and focus to your mind. From deep breathing exercises to mindful observation, these tips can help you manage stress, increase self-awareness, and improve your overall well-being. Useful exercises:")
    st.markdown("- **Deep Breathing:** Find a quiet space, close your eyes, and focus on your breath. Inhale slowly,hold for a few seconds, and exhale slowly. 🌬️")
    st.markdown("- **Body Scan:** Lie down or sit comfortably. Focus your attention on each part of your body, starting with your toes and moving up to your head. 🦶")
    st.markdown(
        "- **Mindful Observation:** Choose an object and observe it with all your senses. Notice its color, shape, texture, and any other details. 🔍")
    st.image("girl.png", caption="Utilising Mindfulness", width=200)

def journaling_section():
    st.subheader("Journaling Prompts 📝")
    st.write("School got you stressed?  Take a moment for yourself with our curated journaling prompts. Reflect on your thoughts, feelings, and experiences with these thought-provoking questions. Journaling can help you gain clarity, reduce stress, and boost your well-being. Here are some thoughtful prompts to get you started:")
    st.markdown("- What are you **grateful** for today? 🙏")
    st.markdown("- What **challenges** did you face, and how did you **overcome** them? 💪")
    st.markdown("- What are your **goals** for the week? 🎯")
    st.image("boy.png", caption="Acheiving Goals", width=200)


mindfulness_section()
journaling_section()

st.title("Conflict Resolutioner AI")
st.write("Navigating work overload, procrastination, friendships, group projects, or misunderstandings with classmates? Our AI-powered conflict resolution assistant is here to help! Describe your situation, and our friendly AI will offer empathetic advice and practical strategies to help you resolve conflicts peacefully and maintain positive relationships.")

conflict_scenarios = [
    "I'm really anxious about an upcoming test.",
    "I'm afraid I'm going to fail my exam.",
    "I have to give a presentation, and I'm terrified of public speaking.",
    "I'm worried I'll mess up my presentation.",
    "I have so much homework, I don't know where to start.",
    "I'm overwhelmed with all the assignments.",
    "I keep procrastinating on my studies, and now I'm behind.",
    "I can't seem to get motivated to do my work.",
    "I'm struggling to understand the material in this class.",
    "I'm confused about the concepts.",
    "Other"
]

conflict_tips = {
    "I'm really anxious about an upcoming test.": [
        "Start to study well in advance, breaking down the material into smaller, manageable chunks.",
        "Take practice tests or quizzes to familiarize yourself with the format and identify areas where you need more focus.",
        "Make sure your notes are clear, concise, and organized. This will make studying more efficient and less stressful.",
        "If you're struggling with the material or have questions, don't hesitate to reach out to your teacher for clarification or extra help.",
        "Practice relaxation techniques like deep breathing or meditation to help manage anxiety leading up to the test."
    ],
    "I'm afraid I'm going to fail my exam": [
        "Remind yourself of what you do know and the areas where you excel. This can boost your confidence.",
        "Imagine yourself taking the exam calmly and confidently, and picture yourself succeeding.",
        "Challenge negative thoughts and replace them with positive affirmations. Believe in your ability to pass.",
        "Talk to friends, family, or a counselor about your fears. Sometimes just expressing your worries can help alleviate them.",
        "Focus on the process of preparing and doing your best, rather than fixating on the possibility of failure."
    ],
    "I have to give a presentation, and I'm terrified of public speaking.": [
        "Knowing your material inside and out will boost your confidence and reduce anxiety.",
        "Rehearse your presentation multiple times, ideally in front of a mirror or a small audience.",
        "Remember that the goal is to share your knowledge and ideas, not to be a perfect performer.",
        "If possible, start with presentations in smaller, less intimidating settings to build your confidence gradually.",
        "Before the presentation, visualize yourself succeeding and practice deep breathing to calm your nerves."
    ],
    "I'm worried I'll mess up my presentation.": [
        "If you're worried about technology glitches or forgetting your place, have a backup plan in place (e.g., printed notes, a USB drive).",
        "It's okay to make minor mistakes. Most people won't even notice, and it doesn't mean you've failed.",
        "Connect with your audience by making eye contact and speaking with passion. This can help you feel more comfortable.",
        "Visual aids can help keep you on track and make your presentation more engaging.",
        " Anticipate potential questions and practice your responses. This will help you feel more prepared and confident."
    ],
    "I have so much homework, I don't know where to start.": [
        "Make a list of all your assignments and prioritize them based on deadlines and importance.",
        "Break down large assignments into smaller, more manageable tasks. This can make the workload seem less daunting.",
        "Plan out your study time, allocating specific blocks for each assignment. Stick to your schedule as much as possible.",
        "Find a quiet study space and eliminate distractions like your phone or social media.",
        "If you're struggling to manage your workload, don't hesitate to ask for help from teachers, classmates, or a tutor."
    ],
    "I'm overwhelmed with all the assignments.": [
        "A visual representation of your schedule can help you see deadlines and manage your time effectively.",
        "Allocate a specific amount of time to each task, even if you don't finish it. This prevents getting bogged down in one assignment.",
        "Tackle the most challenging or unpleasant task first. Getting it out of the way can create momentum.",
        "Instead of long study sessions, try shorter, focused bursts with breaks in between. This can improve concentration.",
        "Focus on completing assignments to the best of your ability within the given time frame, rather than striving for perfection."
    ],
    "I keep procrastinating on my studies, and now I'm behind.": [
        "Try to understand why you're procrastinating. Are you feeling overwhelmed, bored, or anxious?",
        "Start with small, achievable goals to build momentum and avoid feeling discouraged.",
        "Set up a system of rewards for completing tasks. This can help you stay motivated.",
        "Ask a friend or family member to help you stay on track and hold you accountable.",
        "Break down large tasks into smaller, more manageable steps. This can make it easier to get started."
    ],
    "I can't seem to get motivated to do my work.": [
        "Lay out your study materials the night before to reduce the number of decisions you need to make in the morning.",
        "Commit to working on a task for just 5 minutes. Often, this is enough to overcome inertia and get started.",
        "If you're struggling to focus at home, try studying in a library or coffee shop.",
        "Explore apps and tools designed to help with focus and time management, like Forest or Freedom.",
        "Remind yourself of your long-term goals and how completing your studies will help you achieve them."
    ],
    "I'm struggling to understand the material in this class.": [
        "Go back and review the material you're struggling with. Try different learning methods, like reading, watching videos, or creating diagrams.",
        "Don't be afraid to ask your teacher or classmates for clarification on concepts you don't understand.",
        "Look for additional resources like textbooks, online tutorials, or study guides to help you grasp the material.",
        "Studying with classmates can help you learn from each other and gain new perspectives on the material.",
        "The more you practice applying the concepts, the better you'll understand them."
    ],
    "I'm confused about the concepts.": [
        "Explaining the material to someone else, even if it's just a stuffed animal, can help solidify your understanding.",
        "Try to relate new concepts to things you already know. This can make them easier to grasp.",
        "Instead of just rereading notes, test yourself regularly to see what you remember.",
        "Don't be afraid to ask 'dumb' questions. There's no such thing! Asking questions is crucial for learning.",
        "If your teacher's explanation isn't clicking, look for alternative explanations online or in other textbooks."
    ]
}

# Create the dropdown
selected_scenario = st.selectbox("Select a conflict scenario:", conflict_scenarios, key="selectbox2")

user_input = ""  # Initialize user_input

if selected_scenario == "Other":
    user_input = st.text_area("Describe your conflict:")
else:
    user_input = selected_scenario

matched_scenario = None  # Initialize matched_scenario

if selected_scenario == "Other" and user_input:
    # Use TF-IDF to vectorize the scenarios and user input
    vectorizer = TfidfVectorizer()
    scenario_vectors = vectorizer.fit_transform(list(conflict_tips.keys()) + [user_input]) # Corrected: use conflict_tips.keys()

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(scenario_vectors[-1], scenario_vectors[:-1])[0]

    # Find the closest scenario
    closest_scenario_index = similarity_scores.argmax() # Corrected: use argmax() to find the index of the highest similarity
    matched_scenario = list(conflict_tips.keys())[closest_scenario_index] # Corrected: use conflict_tips.keys()

    # Keep the original user input, but use matched_scenario for API
    scenario_for_api = matched_scenario

if st.button("Get Advice"):
    if selected_scenario != "Other": # if not other, use selected scenario.
        if selected_scenario in conflict_tips:
            st.write("**Advice:**")
            for tip in conflict_tips[selected_scenario]:
                st.write(f"- {tip}")
    elif matched_scenario: # if other, check if matching scenario exists
        for tip in conflict_tips[matched_scenario]:
            st.write(f"- {tip}")
    elif selected_scenario == "Other" and not user_input: # if other, and no user input.
        st.write("Please provide a description of your conflict.")
    else:
        st.write("No advice available for the provided scenario.")

def mandela_component(color, brush_size, symmetry_lines):
    print(f"Color: {color}, Brush Size: {brush_size}, Symmetry Lines: {symmetry_lines}") #debugging print statement.
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/konva@8/konva.min.js"></script>
        <style>
            #container {{
                background-color: white;
            }}
            canvas {{
                border: 1px solid black;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <button id="clearButton">Clear</button>
        <script>
            const stage = new Konva.Stage({{
                container: 'container',
                width: 500;,
                height: 500;,
            }});
            const layer = new Konva.Layer();
            stage.add(layer);

            let isDrawing = false;
            let strokeColor = '{color}';
            let strokeWidth = {brush_size};
            let symmetryLines = {symmetry_lines};
            let lastDrawTime = 0;

            stage.on('mousedown touchstart', (e) => {{
                isDrawing = true;
                const pos = stage.getPointerPosition();
                const newLine = new Konva.Line({{
                    points: [pos.x, pos.y],
                    stroke: strokeColor,
                    strokeWidth: strokeWidth,
                    lineCap: 'round',
                    lineJoin: 'round',
                    name: 'userLine'
                }});
                layer.add(newLine);
                currentLine = newLine;
            }});

            stage.on('mousemove touchmove', (e) => {{
                if (!isDrawing) return;
                const currentTime = Date.now();
                if (currentTime - lastDrawTime < 16) return;
                lastDrawTime = currentTime;

                const pos = stage.getPointerPosition();
                const newPoints = currentLine.points().concat([pos.x, pos.y]);
                currentLine.points(newPoints);

                const centerX = stage.width() / 2;
                const centerY = stage.height() / 2;
                const angle = (2 * Math.PI) / symmetryLines;

                layer.getChildren((node) => node.name() === 'symmetryLine' && node.userLineRef === currentLine).forEach((node) => node.destroy());

                for (let i = 1; i < symmetryLines; i++) {{
                    const rotatedPoints = [];
                    for (let j = 0; j < newPoints.length; j += 2) {{
                        const dx = newPoints[j] - centerX;
                        const dy = newPoints[j + 1] - centerY;
                        const rotatedX = dx * Math.cos(angle * i) - dy * Math.sin(angle * i) + centerX;
                        const rotatedY = dx * Math.sin(angle * i) + dy * Math.cos(angle * i) + centerY;
                        rotatedPoints.push(rotatedX, rotatedY);
                    }}
                    const symmetryLine = new Konva.Line({{
                        points: rotatedPoints,
                        stroke: strokeColor,
                        strokeWidth: strokeWidth,
                        lineCap: 'round',
                        lineJoin: 'round',
                        name: 'symmetryLine',
                        userLineRef: currentLine,
                    }});
                    layer.add(symmetryLine);
                }}
                layer.batchDraw();
            }});

            stage.on('mouseup touchend', () => {{
                isDrawing = false;
            }});

            document.getElementById('clearButton').addEventListener('click', function() {{
                layer.getChildren((node) => node.name() === 'userLine' || node.name() === 'symmetryLine').forEach((node) => node.destroy());
                layer.draw();
            }});

            window.addEventListener('message', function(event) {{
                if (event.data.type === 'color_update') {{
                    strokeColor = event.data.color;
                    console.log("Color updated to: ", strokeColor);
                }}
            }});
            console.log("Initial color: ", strokeColor);

        </script>
    </body>
    </html>
    """
    components.html(html_string, height=550)

st.title("Mandala Drawing App")

color = st.color_picker("Choose Color", "#000000")
brush_size = st.slider("Brush Size", 1, 10, 2)
symmetry_lines = st.slider("Symmetry Lines", 2, 20, 8)

mandela_component(color, brush_size, symmetry_lines)

if st.session_state.get('color') != color:
    components.html(f"""
    <script>
        window.dispatchEvent(new MessageEvent('message', {{data: {{type: 'color_update', color: '{color}'}}}}));
        console.log("message dispatched to change color to: ", '{color}');
    </script>
    """, height = 0)
    st.session_state['color'] = color
