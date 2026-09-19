import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
import gradio as gr


# ============================================================
# API / MODEL SETUP
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

tavily = TavilySearch(
    max_results=5
)


# ============================================================
# AGENT STATE
# ============================================================

class MusicAgentState(TypedDict):
    user_request: str
    search_results: str
    recommendations: str


# ============================================================
# ANALYZE PREFERENCES
# ============================================================

def analyze_preferences(state: MusicAgentState):

    prompt = f"""
    Analyze this music recommendation request:

    {state["user_request"]}

    Identify:
    - Mood
    - Genre
    - Language
    - Activity/context
    - Any artist or song preference
    - Any other useful preference

    Keep the analysis concise.
    """

    response = llm.invoke(prompt)

    return {
        "search_results": response.content
    }


# ============================================================
# SEARCH MUSIC
# ============================================================

def search_music(state: MusicAgentState):

    query = f"""
    music recommendations based on:
    {state["user_request"]}

    Find relevant songs, artists, genres and music information.
    Prefer reliable and useful sources.
    """

    results = tavily.invoke({
        "query": query
    })

    return {
        "search_results": str(results)
    }


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(state: MusicAgentState):

    prompt = f"""
    You are a personalized AI music recommendation agent.

    USER REQUEST:
    {state["user_request"]}

    INFORMATION FOUND FROM WEB SEARCH:
    {state["search_results"]}

    Based on the user's request and the available information,
    generate personalized music recommendations.

    Give 5 songs.

    For every song provide:
    1. Song name
    2. Artist
    3. Genre
    4. Why it matches the user's request

    Keep the response clear and easy to read.

    Do not claim that a song is suitable based on information
    that is not supported by the available information.
    """

    response = llm.invoke(prompt)

    return {
        "recommendations": response.content
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(MusicAgentState)

workflow.add_node(
    "analyze_preferences",
    analyze_preferences
)

workflow.add_node(
    "search_music",
    search_music
)

workflow.add_node(
    "generate_recommendations",
    generate_recommendations
)

workflow.add_edge(
    START,
    "analyze_preferences"
)

workflow.add_edge(
    "analyze_preferences",
    "search_music"
)

workflow.add_edge(
    "search_music",
    "generate_recommendations"
)

workflow.add_edge(
    "generate_recommendations",
    END
)

music_agent = workflow.compile()



# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_music(
    mood,
    genre,
    language,
    energy,
    activity,
    artist,
    era,
    description
):

    user_request = f"""
Recommend exactly 5 songs based on these preferences:

Mood: {mood}
Genre: {genre}
Language: {language}
Energy Level: {energy}
Activity: {activity}
Preferred Artist: {artist if artist.strip() else "No specific artist"}
Music Era: {era}

Additional description:
{description if description.strip() else "No additional description provided."}

Return ONLY a clean Markdown table with these columns:

| # | Song | Artist | Genre | Mood |
|---|------|--------|-------|------|

Then give one short sentence below the table.

Do not include:
- Why it fits
- Long explanations
- Sources inside the table
- Preference analysis

Choose songs that genuinely match the user's preferences.
"""

    try:

        result = music_agent.invoke({
            "user_request": user_request,
            "search_results": "",
            "recommendations": ""
        })

        return result["recommendations"]

    except Exception as e:

        return f"""
### ⚠️ Unable to generate recommendations

Please try again.

`{str(e)}`
"""


# ============================================================
# CSS
# ============================================================

custom_css = """

/* ============================================================
   MAIN PAGE
   ============================================================ */

body {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(124, 58, 237, 0.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(59, 130, 246, 0.16),
            transparent 32%
        ),
        #070b18 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: transparent !important;
    padding: 25px !important;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    text-align: center;
    padding: 15px 10px 35px;
}

.hero-icon {
    font-size: 48px;
}

.hero-title {
    margin: 5px 0;

    font-size: 44px;
    font-weight: 850;

    background: linear-gradient(
        90deg,
        #d8b4fe,
        #a78bfa,
        #60a5fa
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #c7cbe0;
    font-size: 19px;
    font-weight: 600;
}

.hero-description {
    color: #858ca3;
    font-size: 14px;
    margin-top: 10px;
}


/* ============================================================
   PREFERENCES CARD
   ============================================================ */

.preferences-card {
    background:
        linear-gradient(
            145deg,
            #121c3d,
            #0b122a
        ) !important;

    border: 1px solid #31477d !important;

    border-radius: 24px !important;

    padding: 28px !important;

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35) !important;
}


/* ============================================================
   PREFERENCE HEADINGS
   ============================================================ */

.preference-title {
    color: #f0f2ff !important;

    font-size: 25px;

    font-weight: 800;

    margin-bottom: 5px;
}

.preference-subtitle {
    color: #8e96af !important;

    font-size: 14px;

    margin-bottom: 22px;
}


/* ============================================================
   INDIVIDUAL PREFERENCE BOX
   ============================================================ */

.preference-box {
    background: #0d1735 !important;

    border: 1px solid #30467d !important;

    border-radius: 18px !important;

    padding: 18px !important;

    min-height: 115px !important;
}

.preference-label {
    color: #edf0ff !important;

    font-size: 16px;

    font-weight: 750;

    margin-bottom: 10px;
}


/* ============================================================
   IMPORTANT:
   REMOVE GRADIO DEFAULT WHITE DROPDOWN CONTAINER
   ============================================================ */

.preference-box .gradio-dropdown {
    background: transparent !important;

    border: none !important;

    box-shadow: none !important;

    padding: 0 !important;

    margin: 0 !important;
}

.preference-box .gradio-dropdown > div {
    background: transparent !important;

    border: none !important;

    box-shadow: none !important;
}


/* ============================================================
   DROPDOWN ITSELF
   ============================================================ */

.preference-box .gradio-dropdown input {
    background: #121d42 !important;

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    font-weight: 750 !important;

    font-size: 16px !important;

    border: 1px solid #5269a3 !important;

    border-radius: 12px !important;

    box-shadow: none !important;

    min-height: 42px !important;
}

.preference-box .gradio-dropdown input::placeholder {
    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    opacity: 1 !important;
}


/* ============================================================
   DROPDOWN ARROW
   ============================================================ */

.preference-box .gradio-dropdown svg {
    color: #ffffff !important;

    fill: #ffffff !important;

    stroke: #ffffff !important;
}


/* ============================================================
   DROPDOWN OPEN MENU
   ============================================================ */

.gradio-dropdown [role="listbox"],
.gradio-dropdown .options,
.gradio-dropdown .options-container {
    background: #111a38 !important;

    color: #ffffff !important;

    border: 1px solid #5269a3 !important;

    border-radius: 12px !important;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.65) !important;
}


/* ============================================================
   DROPDOWN OPTIONS
   ============================================================ */

.gradio-dropdown [role="option"],
.gradio-dropdown .options li {
    background: #111a38 !important;

    color: #ffffff !important;

    font-size: 15px !important;

    font-weight: 700 !important;

    padding: 11px 14px !important;
}

.gradio-dropdown [role="option"]:hover,
.gradio-dropdown .options li:hover {
    background: #283867 !important;

    color: #ffffff !important;
}


/* selected option */

.gradio-dropdown [role="option"][aria-selected="true"] {
    background: #3d5188 !important;

    color: #ffffff !important;

    font-weight: 800 !important;
}


/* ============================================================
   TEXT INPUT
   ============================================================ */

.preference-box input[type="text"] {
    background: #121d42 !important;

    color: #ffffff !important;

    border: 1px solid #5269a3 !important;

    border-radius: 12px !important;

    font-weight: 600 !important;

    padding: 12px !important;
}

.preference-box input[type="text"]::placeholder {
    color: #7d87a5 !important;
}


/* ============================================================
   DESCRIPTION BOX
   ============================================================ */

textarea {
    background: #0e1835 !important;

    color: #ffffff !important;

    border: 1px solid #40558c !important;

    border-radius: 14px !important;

    padding: 15px !important;

    font-size: 14px !important;
}

textarea::placeholder {
    color: #737d9a !important;
}


/* ============================================================
   GET RECOMMENDATIONS BUTTON
   ============================================================ */

#get-recommendations {
    height: 60px !important;

    border-radius: 15px !important;

    border: none !important;

    background:
        linear-gradient(
            90deg,
            #a855f7,
            #7c3aed,
            #3b82f6
        ) !important;

    color: #ffffff !important;

    font-size: 18px !important;

    font-weight: 800 !important;

    box-shadow:
        0 12px 35px rgba(99,70,220,0.35) !important;

    margin-top: 8px !important;
}

#get-recommendations:hover {
    transform: translateY(-2px);

    box-shadow:
        0 16px 42px rgba(99,70,220,0.50) !important;
}


/* ============================================================
   QUICK EXAMPLES
   ============================================================ */

.quick-card {
    margin-top: 22px;

    padding: 20px;

    border-radius: 18px;

    background: #0d1735 !important;

    border: 1px solid #30467d !important;
}

.quick-title {
    color: #edf0ff;

    font-size: 16px;

    font-weight: 750;
}

.quick-subtitle {
    color: #7f88a2;

    font-size: 13px;

    margin: 5px 0 14px;
}

.quick-btn {
    background: #121d42 !important;

    color: #ffffff !important;

    border: 1px solid #3c5188 !important;

    border-radius: 12px !important;

    font-weight: 700 !important;
}

.quick-btn:hover {
    background: #1d2b55 !important;

    border-color: #8b5cf6 !important;
}


/* ============================================================
   RECOMMENDATION TITLE
   ============================================================ */

.output-title {
    margin-top: 35px;

    margin-bottom: 15px;

    color: #4c3a91 !important;

    font-size: 25px;

    font-weight: 800;
}


/* ============================================================
   RECOMMENDATION BOX
   ============================================================ */

#recommendation-output {
    background:
        linear-gradient(
            145deg,
            #18213f,
            #10172d
        ) !important;

    border: 1px solid #394d82 !important;

    border-radius: 22px !important;

    padding: 28px !important;

    min-height: 300px !important;

    color: #f0f2ff !important;

    box-shadow:
        0 18px 55px rgba(0,0,0,0.35) !important;
}


/* ============================================================
   FIX PLACEHOLDER TEXT
   ============================================================ */

#recommendation-output h1,
#recommendation-output h2,
#recommendation-output h3,
#recommendation-output h4,
#recommendation-output p,
#recommendation-output span,
#recommendation-output strong {
    color: #f0f2ff !important;
}


/* ============================================================
   RECOMMENDATION TABLE
   ============================================================ */

#recommendation-output table {
    width: 100% !important;

    border-collapse: separate !important;

    border-spacing: 0 !important;

    border: 1px solid #3b4f80 !important;

    border-radius: 14px !important;

    overflow: hidden !important;
}

#recommendation-output th {
    background: #25345d !important;

    color: #ffffff !important;

    font-weight: 800 !important;

    font-size: 15px !important;

    padding: 15px !important;

    border-bottom: 1px solid #435a91 !important;
}

#recommendation-output td {
    background: #121c38 !important;

    color: #e8eaf5 !important;

    font-size: 14px !important;

    padding: 15px !important;

    border-bottom: 1px solid #2d3d67 !important;
}

#recommendation-output tr:last-child td {
    border-bottom: none !important;
}

#recommendation-output tr:hover td {
    background: #19264a !important;
}


/* ============================================================
   TECH STRIP
   ============================================================ */

.tech-strip {
    margin-top: 25px;

    padding: 15px;

    text-align: center;

    border-radius: 14px;

    background: rgba(99,102,241,0.06);

    border: 1px solid rgba(99,102,241,0.18);

    color: #858da7;

    font-size: 13px;
}

.footer {
    text-align: center;

    padding: 20px;

    color: #626b84;

    font-size: 12px;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .hero-title {
        font-size: 32px;
    }

    .preferences-card {
        padding: 18px !important;
    }

}


/* ============================================================
   HIDE GRADIO DEFAULT FOOTER
   ============================================================ */

footer {
    display: none !important;
}

"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="AI Music Recommendation Agent",
    css=custom_css,
    theme=gr.themes.Base()
) as demo:


    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML("""
    <div class="hero">

        <div class="hero-icon">🎵</div>

        <h1 class="hero-title">
            AI MUSIC RECOMMENDER
        </h1>

        <div class="hero-subtitle">
            Personalized Music Suggestion Agent
        </div>

        <div class="hero-description">
            Tell us your preferences and let AI discover
            music that matches your mood, activity and vibe.
        </div>

    </div>
    """)


    # --------------------------------------------------------
    # PREFERENCES
    # --------------------------------------------------------

    with gr.Column(elem_classes="preferences-card"):

        gr.HTML("""
        <div class="preference-title">
            🎛️ Music Preferences
        </div>

        <div class="preference-subtitle">
            Set your preferences below or describe what you're
            in the mood for.
        </div>
        """)


        # ====================================================
        # ROW 1
        # ====================================================

        with gr.Row():

            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    😊 Mood
                </div>
                """)

                mood = gr.Dropdown(
                    choices=[
                        "Happy",
                        "Sad",
                        "Relaxed",
                        "Romantic",
                        "Energetic",
                        "Motivational",
                        "Peaceful",
                        "Nostalgic",
                        "Focused",
                        "Any"
                    ],
                    value="Happy",
                    show_label=False,
                    container=False
                )


            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    🎵 Genre
                </div>
                """)

                genre = gr.Dropdown(
                    choices=[
                        "Pop",
                        "Bollywood",
                        "Rock",
                        "Hip-Hop",
                        "R&B",
                        "Electronic",
                        "Classical",
                        "Lo-fi",
                        "Indie",
                        "Jazz",
                        "Acoustic",
                        "Any"
                    ],
                    value="Pop",
                    show_label=False,
                    container=False
                )


        # ====================================================
        # ROW 2
        # ====================================================

        with gr.Row():

            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    🌐 Language
                </div>
                """)

                language = gr.Dropdown(
                    choices=[
                        "English",
                        "Hindi",
                        "Marathi",
                        "Punjabi",
                        "Tamil",
                        "Telugu",
                        "Bengali",
                        "Korean",
                        "Spanish",
                        "Any"
                    ],
                    value="English",
                    show_label=False,
                    container=False
                )


            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    ⚡ Energy Level
                </div>
                """)

                energy = gr.Dropdown(
                    choices=[
                        "💤 Very Low",
                        "🌿 Low",
                        "🙂 Medium",
                        "⚡ High",
                        "🔥 Very High"
                    ],
                    value="🙂 Medium",
                    show_label=False,
                    container=False
                )


        # ====================================================
        # ROW 3
        # ====================================================

        with gr.Row():

            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    🏃 Activity
                </div>
                """)

                activity = gr.Dropdown(
                    choices=[
                        "Studying",
                        "Working",
                        "Workout",
                        "Relaxing",
                        "Traveling",
                        "Party",
                        "Driving",
                        "Sleeping",
                        "Meditation",
                        "Any"
                    ],
                    value="Studying",
                    show_label=False,
                    container=False
                )


            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    🎤 Artist
                    <span style="color:#727991;">
                        (Optional)
                    </span>
                </div>
                """)

                artist = gr.Textbox(
                    placeholder="e.g. Arijit Singh, Taylor Swift...",
                    show_label=False,
                    container=False
                )


        # ====================================================
        # ROW 4 — ERA
        # ====================================================

        with gr.Row():

            with gr.Column(elem_classes="preference-box"):

                gr.HTML("""
                <div class="preference-label">
                    🕰️ Music Era
                </div>
                """)

                era = gr.Dropdown(
                    choices=[
                        "Any Era",
                        "Recent (2020s)",
                        "2010s",
                        "2000s",
                        "1990s",
                        "1980s",
                        "1970s",
                        "1960s & Earlier"
                    ],
                    value="Any Era",
                    show_label=False,
                    container=False
                )


        # ====================================================
        # DESCRIPTION
        # ====================================================

        gr.HTML("""
        <div style="
            margin-top:20px;
            margin-bottom:8px;
            color:#e5e7f5;
            font-size:15px;
            font-weight:700;
        ">
            ✍️ Describe Your Vibe
            <span style="
                color:#727991;
                font-weight:400;
            ">
                (Optional)
            </span>
        </div>

        <div style="
            color:#777f99;
            font-size:13px;
            margin-bottom:8px;
        ">
            Tell the agent anything else about what you're feeling,
            or leave it empty.
        </div>
        """)

        description = gr.Textbox(
            placeholder="e.g. I want calm music for a rainy evening...",
            lines=4,
            max_lines=5,
            show_label=False
        )


        # ====================================================
        # BUTTON
        # ====================================================

        recommend_button = gr.Button(
            "🎧  Get Personalized Recommendations  →",
            variant="primary",
            elem_id="get-recommendations"
        )


    # ========================================================
    # QUICK EXAMPLES
    # ========================================================

    with gr.Column(elem_classes="quick-card"):

        gr.HTML("""
        <div class="quick-title">
            💡 Quick Examples
        </div>

        <div class="quick-subtitle">
            Click an example to automatically fill your preferences.
        </div>
        """)

        with gr.Row():

            relaxing_btn = gr.Button(
                "🌙 Relaxing",
                elem_classes="quick-btn"
            )

            study_btn = gr.Button(
                "📚 Study",
                elem_classes="quick-btn"
            )

            workout_btn = gr.Button(
                "⚡ Workout",
                elem_classes="quick-btn"
            )

            romantic_btn = gr.Button(
                "❤️ Romantic",
                elem_classes="quick-btn"
            )

            party_btn = gr.Button(
                "🎉 Party",
                elem_classes="quick-btn"
            )


    # ========================================================
    # OUTPUT TITLE
    # ========================================================

    gr.HTML("""
    <div class="output-title">
        ✨ Personalized Recommendations
    </div>
    """)


    # ========================================================
    # OUTPUT BOX
    # ========================================================

    output = gr.Markdown(
        value="""
<div style="
    text-align:center;
    padding:55px 20px;
">

<h3 style="
    color:#eef0ff !important;
    font-size:21px;
">
    🎶 Your recommendations will appear here
</h3>

<p style="
    color:#9ca5c0 !important;
    font-size:15px;
">
    Choose your preferences above and click
    <strong style="color:#ffffff !important;">
        Get Personalized Recommendations
    </strong>.
</p>

</div>
        """,
        elem_id="recommendation-output"
    )


    
    # ========================================================
    # QUICK EXAMPLE ACTIONS
    # ========================================================

    relaxing_btn.click(
        lambda: (
            "Relaxed",
            "Lo-fi",
            "English",
            "🌿 Low",
            "Relaxing",
            "",
            "Any Era",
            "I want calm and peaceful music for a quiet evening."
        ),
        outputs=[
            mood,
            genre,
            language,
            energy,
            activity,
            artist,
            era,
            description
        ]
    )


    study_btn.click(
        lambda: (
            "Focused",
            "Lo-fi",
            "English",
            "🌿 Low",
            "Studying",
            "",
            "Any Era",
            "I want calm music that helps me concentrate."
        ),
        outputs=[
            mood,
            genre,
            language,
            energy,
            activity,
            artist,
            era,
            description
        ]
    )


    workout_btn.click(
        lambda: (
            "Energetic",
            "Pop",
            "English",
            "🔥 Very High",
            "Workout",
            "",
            "Recent (2020s)",
            "I want energetic and motivating songs for my workout."
        ),
        outputs=[
            mood,
            genre,
            language,
            energy,
            activity,
            artist,
            era,
            description
        ]
    )


    romantic_btn.click(
        lambda: (
            "Romantic",
            "Bollywood",
            "Hindi",
            "🙂 Medium",
            "Relaxing",
            "",
            "2010s",
            "I want romantic Hindi songs for a date night."
        ),
        outputs=[
            mood,
            genre,
            language,
            energy,
            activity,
            artist,
            era,
            description
        ]
    )


    party_btn.click(
        lambda: (
            "Happy",
            "Pop",
            "English",
            "🔥 Very High",
            "Party",
            "",
            "Recent (2020s)",
            "I want fun upbeat songs for a party with friends."
        ),
        outputs=[
            mood,
            genre,
            language,
            energy,
            activity,
            artist,
            era,
            description
        ]
    )


    # ========================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================

    recommend_button.click(
        fn=recommend_music,
        inputs=[
            mood,
            genre,
            language,
            energy,
            activity,
            artist,
            era,
            description
        ],
        outputs=output
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
