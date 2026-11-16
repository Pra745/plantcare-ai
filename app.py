import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root {
  --bg:#F2F0E7; --surface:#FFFEFA; --surface-alt:#E7E9DC;
  --forest:#2B5B4B; --forest-deep:#214838; --moss:#66836A;
  --sage:#A9B99A; --clay:#C87552; --ink:#20362E;
  --muted:#65766B; --line:#D5DCCE;
}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:var(--bg);color:var(--ink);}
[data-testid="stHeader"] {background:rgba(242,240,231,.96);}
[data-testid="stSidebar"] {background:var(--forest);border-right:1px solid var(--forest-deep);}
[data-testid="stSidebar"] * {color:#F7F5ED;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.48rem .35rem;font-size:1.02rem;}
.brand {font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:700;color:#fff;margin:.2rem 0;}
.brand-subtitle {color:#D8E2D5;font-size:.9rem;line-height:1.5;}
h1,h2 {font-family:'Playfair Display',serif!important;color:var(--forest)!important;}
h3,h4 {color:var(--forest)!important;}
.hero {background:linear-gradient(125deg,#2B5B4B,#47765D);padding:2.5rem 2.4rem;border-radius:24px;color:white;margin-bottom:1.5rem;}
.hero h1 {color:white!important;font-size:3rem;line-height:1.12;margin:.7rem 0;}
.hero p {color:#E7EBDD;font-size:1.08rem;max-width:760px;line-height:1.7;}
.kicker {color:var(--clay);font-weight:700;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;margin:.8rem 0;}
.soft-card {background:var(--surface);border:1px solid var(--line);border-radius:17px;padding:1.15rem 1.25rem;margin:.55rem 0;}
.soft-card p {color:var(--muted);line-height:1.65;margin-bottom:.1rem;}
/* Equal-height homepage feature cards */
.home-feature-card {
    height: 255px;
    min-height: 255px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: flex-start !important;
    gap: 12px;
}

.home-feature-card h3 {
    margin: 8px 0 4px !important;
}

.home-feature-card p {
    margin: 0 !important;
    padding: 0 !important;
}

/* Consistent button dimensions */
div.stButton > button {
    min-height: 3rem;
    height: 3rem;
    width: 100%;
    border-radius: 10px;
}
.feature-icon {font-size:1.7rem;margin-bottom:.35rem;}
.result-banner {background:var(--forest);padding:1.7rem 1.8rem;border-radius:19px;color:white;margin:.8rem 0;}
.result-label {font-size:.7rem;letter-spacing:.15em;font-weight:700;color:#D8E2D5;text-transform:uppercase;}
.result-name {font-family:'Playfair Display',serif;font-size:2.5rem;color:white;margin:.4rem 0;}
.result-copy {color:#E7EBDD;font-size:1rem;}
.callout {background:#E3EBDD;border:1px solid #C8D7C0;border-left:4px solid var(--moss);padding:1rem 1.2rem;border-radius:12px;margin:1rem 0;}
div.stButton>button[kind="primary"] {background:var(--forest);border:0;color:white;border-radius:11px;min-height:3rem;font-weight:700;}
div.stButton>button[kind="primary"]:hover {background:var(--forest-deep);color:white;}
div.stButton>button {border-radius:10px;}
[data-testid="stMetric"] {background:#E7E9DC;border:1px solid #D5DCCE;padding:1rem;border-radius:15px;}
.footer {border-top:1px solid var(--line);margin-top:3rem;padding:1.2rem;text-align:center;color:var(--muted);font-size:.82rem;line-height:1.7;}
.empty-preview {height:290px;border:1px dashed #B9C6B2;border-radius:18px;background:#E7E9DC;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#65766B;text-align:center;}
.small-note {font-size:.82rem;color:var(--muted);}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CNN_PATH = MODEL_DIR / "final_plant_cnn.keras"
MOBILENET_PATH = MODEL_DIR / "best_mobilenetv2.keras"
CLASS_PATH = MODEL_DIR / "class_names.json"
EXPECTED_CLASSES = 38

# Values are displayed as reported/verified, not inferred from confidence scores.
CNN_TEST_ACCURACY = 0.924002468585968
MOBILENET_TEST_ACCURACY = None  # Person 2's test-evaluation report was not supplied with the model file.

# ============================================================
# MODEL + LABEL UTILITIES
# ============================================================
@st.cache_data(show_spinner=False)
def load_class_names(path_string):
    with open(path_string, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        try:
            labels = [labels[k] for k in sorted(labels, key=lambda item: int(item))]
        except (TypeError, ValueError):
            labels = list(labels.values())
    labels = [str(x) for x in labels]
    if len(labels) != EXPECTED_CLASSES:
        raise ValueError(f"Expected 38 class labels; found {len(labels)}.")
    return labels

@st.cache_resource(show_spinner=False)
def load_keras_model(path_string):
    import tensorflow as tf
    return tf.keras.models.load_model(path_string, compile=False)

def human_label(raw_label):
    crop, sep, disease = raw_label.partition("___")
    crop = crop.replace("_", " ").replace(",", ",").strip()
    disease = disease.replace("_", " ").strip() if sep else "Unknown class"
    disease = disease.replace("  ", " ")
    if disease.lower() == "healthy":
        return crop, "Healthy"
    return crop, disease

def scores_to_probabilities(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("The model returned invalid numeric outputs.")
    if values.min() >= 0 and values.max() <= 1 and np.isclose(values.sum(), 1.0, atol=0.03):
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()

def predict_with_model(image, model_key, labels):
    if model_key == "Custom CNN":
        model_path = CNN_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Custom CNN file not found: {model_path}")
    elif model_key == "MobileNetV2":
        model_path = MOBILENET_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"MobileNetV2 file not found: {model_path}")
    else:
        raise ValueError("Unknown model selection.")

    model = load_keras_model(str(model_path))
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)

    rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    batch = np.asarray(rgb, dtype=np.float32)

    if model_key == "Custom CNN":
        # Matches Person 1's saved training summary: RGB values scaled to [0, 1].
        batch = batch / 255.0
    else:
        # Matches MobileNetV2's standard preprocess_input convention: [0,255] -> [-1,1].
        import tensorflow as tf
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    batch = np.expand_dims(batch, axis=0)
    raw = model.predict(batch, verbose=0).reshape(-1)

    if len(raw) != len(labels):
        raise ValueError(
            f"{model_key} returned {len(raw)} outputs, but class_names.json has {len(labels)} labels. "
            "The model and label order must match."
        )

    probabilities = scores_to_probabilities(raw)
    order = np.argsort(probabilities)[::-1][:3]
    return [(labels[int(i)], float(probabilities[int(i)])) for i in order]

def model_file_status():
    return {
        "Custom CNN": CNN_PATH.exists(),
        "MobileNetV2": MOBILENET_PATH.exists(),
    }

# ============================================================
# GEMINI — RETRIES + MODEL FALLBACK
# ============================================================
def get_gemini_key():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = os.getenv("GEMINI_API_KEY")
    if not key or key == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            "Gemini key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml."
        )
    return key

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_gemini_text(prompt):
    """Cache identical prompts for one hour to avoid repeat API calls."""
    from google import genai

    client = genai.Client(api_key=get_gemini_key())
    last_error = None
    # Flash-Lite is the low-latency choice; Flash is a fallback if the first model is unavailable.
    for model_id in ("gemini-3.5-flash-lite", "gemini-3.8-flash"):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={"max_output_tokens": 220, "temperature": 0.35},
            )
            answer = (response.text or "").strip()
            if answer:
                return answer
            last_error = RuntimeError("The AI service returned an empty answer.")
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            # Try the fallback model only for unavailable/unsupported primary model errors.
            if not any(token in message for token in ("404", "not found", "unsupported", "503", "unavailable", "429", "resource_exhausted", "500")):
                break
    raise RuntimeError(
        "The AI service is temporarily unavailable or this API key cannot access the selected model. "
        "Your CNN prediction is still available. Please try again shortly."
    ) from last_error


def generate_ai_explanation(raw_label, model_name):
    crop, disease = human_label(raw_label)
    prompt = f"""Explain this plant classifier result to a beginner in simple everyday English.

Plant: {crop}
Prediction: {disease}

Write no more than 80 words using exactly these three short headings:
What it means
What you might notice
What to do next

Use short sentences and familiar words. Avoid scientific jargon and abbreviations.
Do not say you inspected the photo or that the prediction is certain.
Give safe observation steps only. Do not recommend chemicals or pesticides.
The classifier used was {model_name}.
"""
    return _cached_gemini_text(prompt)


def generate_followup_answer(raw_label, question):
    crop, disease = human_label(raw_label)
    prompt = f"""You are a friendly plant-care learning assistant.
The image classifier predicted {crop} — {disease}.
Answer this selected beginner question: {question}
Use simple everyday English in 2–4 short sentences. Be practical and calm.
Do not claim the prediction is confirmed. Do not recommend pesticides or chemicals.
"""
    return _cached_gemini_text(prompt)



# ============================================================
# NEW FEATURES: PDF REPORT, CROP PROFILES, PLANTCARE GLOSSARY
# ============================================================
def create_prediction_pdf(leaf_image, crop, disease, model_name, confidence, ai_explanation=""):
    """Build a downloadable PDF report for the current prediction."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage

    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PlantCareTitle", parent=styles["Title"],
        textColor=colors.HexColor("#173F35"), fontSize=23, leading=28,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="PlantCareHeading", parent=styles["Heading2"],
        textColor=colors.HexColor("#173F35"), spaceBefore=10, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="PlantCareBody", parent=styles["BodyText"], leading=15, spaceAfter=6
    ))

    story = [
        Paragraph("PlantCare AI", styles["PlantCareTitle"]),
        Paragraph("Plant Health Prediction Report", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["BodyText"]),
        Spacer(1, 8),
    ]
    if leaf_image is not None:
        image_buffer = BytesIO()
        leaf_image.convert("RGB").save(image_buffer, format="JPEG", quality=88)
        image_buffer.seek(0)
        report_image = PDFImage(image_buffer, width=100*mm, height=67*mm)
        report_image.hAlign = "CENTER"
        story.extend([report_image, Spacer(1, 10)])

    details = [
        ["Crop", crop],
        ["Predicted class", disease],
        ["Model used", model_name],
        ["Model confidence", f"{confidence:.1%}"],
    ]
    table = Table(details, colWidths=[48*mm, 112*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E7E9DC")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#20362E")),
        ("GRID", (0,0), (-1,-1), .5, colors.HexColor("#D5DCCE")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.extend([table, Spacer(1, 10)])
    story.append(Paragraph("Understanding this result", styles["PlantCareHeading"]))
    story.append(Paragraph(
        "This report records the output of an educational image-classification model. "
        "The confidence score is the model's score for its selected class; it is not the "
        "same as test accuracy and does not guarantee that the prediction is correct.",
        styles["PlantCareBody"]
    ))
    if ai_explanation:
        safe_text = (ai_explanation.replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;").replace("\n", "<br/>"))
        story.extend([
            Paragraph("AI-generated explanation", styles["PlantCareHeading"]),
            Paragraph(safe_text, styles["PlantCareBody"]),
        ])
    story.extend([
        Paragraph("Suggested next step", styles["PlantCareHeading"]),
        Paragraph(
            "Observe the whole plant and note whether symptoms are spreading. "
            "For uncertain or consequential crop-health concerns, consult a qualified "
            "local agricultural professional or extension worker.",
            styles["PlantCareBody"]
        ),
        Spacer(1, 8),
        Paragraph(
            "PlantCare AI · Student project · Educational information, not a confirmed diagnosis.",
            styles["Italic"]
        ),
    ])
    document.build(story)
    output.seek(0)
    return output.getvalue()


CROP_CARE_PROFILES = {
    "Apple": {
        "light": "Choose a site that receives several hours of direct sunlight. Good light supports flowering and fruit development, while open spacing helps leaves dry after rain. In very hot locations, young trees may need protection while establishing.",
        "water": "Water newly planted trees deeply during dry spells while their roots establish. Mature trees need extra attention during prolonged dry weather and fruit development. Water the root area rather than repeatedly wetting the leaves, and avoid soil that stays saturated.",
        "soil": "Apple trees generally prefer deep, fertile, well-drained soil. Before planting, check drainage and consider a soil test; soil pH, rootstock and variety influence suitability. Avoid planting where water regularly collects around the roots.",
        "routine": "Inspect both sides of leaves and developing fruit every week. Remove fallen or visibly diseased material where appropriate, keep the base of the tree reasonably clear, and prune using clean tools during the locally recommended season. Follow local orchard advice for thinning and tree structure.",
        "monitor": "Watch for olive-green or dark leaf spots, premature leaf drop, fruit blemishes, cankers or unusual shoot dieback. Record when symptoms first appear and whether they affect one branch or the whole tree. Similar marks can have different causes, so use the model result as a clue rather than confirmation.",
        "seasonal": "During active growth, monitor new leaves and fruit regularly. After harvest, review tree health and clear fallen debris where appropriate. Pruning timing differs by climate and variety, so use regional orchard guidance rather than a universal calendar.",
        "classes": "Apple scab, black rot, cedar-apple rust and healthy apple."
    },
    "Blueberry": {
        "light": "Blueberries generally perform best in a sunny position, although protection from extreme afternoon heat can help in very hot climates. Allow enough room around bushes for air movement and access during harvesting.",
        "water": "Blueberries have relatively shallow roots and benefit from consistent moisture, especially while establishing and when berries are developing. Check the soil regularly during dry weather. Water thoroughly but allow excess water to drain; both drought and waterlogged roots can cause stress.",
        "soil": "Acidic soil is especially important for blueberries. Before planting, check soil pH and use a growing medium suitable for acid-loving plants if needed. Good drainage and organic matter help create a suitable root environment; ordinary garden soil may not be acidic enough.",
        "routine": "Keep a moisture-retaining mulch over the root zone, leaving space around the stems. Check leaves for unusual yellowing and inspect berries for damage. Pruning needs depend on bush age and variety; remove old or weak canes according to local blueberry-growing guidance.",
        "monitor": "Look for yellow leaves, weak new growth, leaf spots, shrivelling berries or branches that die back. Yellowing can be linked to soil pH or nutrient availability as well as disease, so check growing conditions before assuming infection.",
        "seasonal": "During flowering and fruiting, keep moisture steady and protect the shallow root zone from drying. After harvest, review plant structure and remove damaged material when appropriate. Adjust mulch and watering as seasons change.",
        "classes": "Healthy blueberry."
    },
    "Cherry": {
        "light": "Choose a sunny, open location. Good sunlight supports flowering and fruit development, while adequate spacing helps reduce prolonged dampness around foliage. Consider local chill and temperature requirements when selecting a variety.",
        "water": "Water young trees during dry periods, especially while roots are establishing. Mature trees may need additional water during prolonged dry weather and fruit development. Water the root area and maintain drainage rather than keeping the soil constantly wet.",
        "soil": "Cherry trees generally need well-drained soil. Heavy, poorly drained ground can stress roots, so assess the site before planting. Soil depth, pH and variety suitability should be checked against local horticultural guidance.",
        "routine": "Inspect leaves, shoots and fruit for spots, powdery growth, cracking or dieback. Remove damaged fallen material where appropriate and use clean pruning tools. Pruning timing and intensity depend on cherry type and local disease pressure.",
        "monitor": "Check for pale powder-like patches, leaf spots, premature leaf drop, damaged fruit and branch dieback. Note whether symptoms appear after wet weather or are limited to particular branches. A photograph cannot reliably distinguish every cause.",
        "seasonal": "During flowering and fruiting, watch for weather-related stress and fruit damage. After harvest, inspect the canopy and plan any maintenance using regional recommendations. Avoid pruning at unsuitable times for the variety and climate.",
        "classes": "Powdery mildew and healthy cherry."
    },
    "Corn": {
        "light": "Corn needs an open, sunny field for strong growth. Avoid excessive shading from nearby structures or taller crops. Planting density should follow local recommendations so the crop receives light and has room for airflow.",
        "water": "Maintain suitable soil moisture during establishment, flowering and grain filling. Water needs rise during hot or dry weather, but prolonged waterlogging can damage roots. Check field conditions at more than one point because soil moisture can vary across a plot.",
        "soil": "Fertile, well-drained soil supports healthy root and plant development. Use a soil test and local crop recommendations to guide nutrient management rather than applying fertiliser by guesswork. Avoid compaction where possible.",
        "routine": "Scout the field regularly in a consistent pattern. Inspect lower, middle and upper leaves, and note whether symptoms begin on older leaves or new growth. Record affected areas and whether damage is spreading between plants.",
        "monitor": "Look for elongated grey-green or tan lesions, rust-coloured pustules, small leaf spots, lodging or uneven growth. Compare several plants rather than relying on one leaf. Record the field location and recent rainfall or irrigation.",
        "seasonal": "Check establishment and early leaf development first, then increase scouting around periods of rapid growth and reproductive development. After heavy rain or humid weather, inspect leaves for new spots and document any spread.",
        "classes": "Common rust, Northern Leaf Blight, Cercospora leaf spot and healthy corn."
    },
    "Grape": {
        "light": "Grapevines generally need a sunny site. Choose a position with good air circulation and enough space for the intended trellis or support system. Dense shade can reduce fruit quality and keep leaves damp for longer.",
        "water": "Water young vines consistently while they establish. For mature vines, watering needs depend on soil, climate, root depth and growth stage. Avoid frequent shallow watering and prevent the root zone from remaining saturated.",
        "soil": "Well-drained soil is important for grape roots. Soil type and vine variety influence vigour and water needs. Before planting, consider drainage, support structures and local recommendations for the chosen grape variety.",
        "routine": "Train shoots to the support system and manage canopy density according to local viticulture guidance. Inspect upper and lower leaf surfaces, stems and bunches. Clean tools between plants when disease is suspected and remove affected debris where appropriate.",
        "monitor": "Watch for dark spots, leaf discolouration, powdery or downy-looking growth, shrivelling berries and unusual cane lesions. Note whether symptoms follow humid weather and whether they occur throughout the canopy or in isolated areas.",
        "seasonal": "Inspect new growth during the growing season and pay close attention around flowering and fruit development. After harvest, review vine structure and follow local advice on pruning and sanitation.",
        "classes": "Black rot, Esca, leaf blight and healthy grape."
    },
    "Orange": {
        "light": "Citrus generally needs a warm, sunny position. Protect trees from damaging cold and strong drying winds where possible. Container-grown citrus may need to be moved or sheltered when temperatures fall outside the variety's tolerance.",
        "water": "Water deeply when the root zone begins to dry, adjusting frequency for pot size, soil and weather. Young trees need closer monitoring than established trees. Ensure excess water can drain and avoid leaving container roots standing in water.",
        "soil": "Use a well-drained growing medium. Citrus roots are sensitive to prolonged waterlogging, while compacted soil can restrict root growth. For container trees, use a suitable citrus or free-draining potting mix and a pot with drainage holes.",
        "routine": "Inspect new shoots and the undersides of leaves for insects, sticky residue, curling or unusual yellowing. Keep the area around the trunk clear and avoid damaging roots during cultivation. Follow local citrus guidance for pruning and nutrition.",
        "monitor": "Look for mottled yellow patterns, leaf curling, twig dieback, misshapen fruit or uneven growth. Some serious citrus diseases require official local assessment; do not rely on a photo prediction to confirm or rule them out.",
        "seasonal": "Check watering more often during hot weather and reduce it when growth slows or conditions are cooler. Watch new flushes for pests and unusual patterns. Follow regional guidance for citrus nutrition and any regulated disease concerns.",
        "classes": "Huanglongbing (citrus greening) and healthy orange."
    },
    "Peach": {
        "light": "Peach trees generally need a sunny location with good airflow. Select a variety suited to local winter and summer conditions. Avoid cramped sites where foliage remains damp for long periods after rain.",
        "water": "Water young trees during establishment and maintain suitable moisture during fruit development. Adjust watering for rainfall and soil type. Avoid repeated wetting of leaves and do not allow roots to remain in saturated soil.",
        "soil": "Choose well-drained soil and assess the site before planting. Soil depth, pH and rootstock influence tree performance. Local orchard guidance can help match variety and management to the growing region.",
        "routine": "Inspect leaves, shoots and fruit regularly. Remove damaged material when appropriate and use clean tools for pruning. Maintain a manageable canopy without removing excessive healthy growth.",
        "monitor": "Look for small dark or water-soaked-looking leaf spots, holes where damaged tissue has fallen away, shoot lesions and fruit blemishes. Record whether symptoms appear after rain and whether new leaves are affected.",
        "seasonal": "Monitor leaf and fruit development through the growing season. During wet periods, inspect foliage more frequently. Follow local recommendations for pruning, fruit thinning and orchard sanitation.",
        "classes": "Bacterial spot and healthy peach."
    },
    "Pepper": {
        "light": "Peppers need a warm, bright growing location. Most perform well with plenty of sunlight, but extreme heat can affect flowering and fruit set. Provide enough spacing for light and airflow around the plants.",
        "water": "Keep soil moisture reasonably even, especially during flowering and fruit development. Water at the base of the plant and allow excess water to drain. Large swings between very dry and very wet conditions can stress plants.",
        "soil": "Use a fertile, well-drained growing medium. In containers, choose a pot with drainage holes and enough volume for the plant. Avoid excessive fertiliser; follow crop and product guidance for nutrient needs.",
        "routine": "Inspect both sides of leaves and check stems and fruit for spots, curling or insect activity. Support heavily loaded plants if needed. Remove fallen debris and avoid handling wet plants when disease is suspected.",
        "monitor": "Look for dark leaf spots, yellowing, curling, wilting or blemishes on fruit. Check whether symptoms are limited to older leaves or also appear on new growth. Heat, watering stress and pests can resemble disease symptoms.",
        "seasonal": "During flowering and fruiting, pay close attention to water consistency and heat stress. Remove damaged leaves only when appropriate and avoid stripping the plant of healthy foliage.",
        "classes": "Bacterial spot and healthy bell pepper."
    },
    "Potato": {
        "light": "Potatoes grow best in an open, sunny location. Choose a site with good drainage and avoid planting repeatedly in the same ground where crop rotation is practical. Follow local recommendations for planting time and variety.",
        "water": "Maintain reasonably even soil moisture, particularly during tuber development. Water needs change with weather and soil type. Avoid waterlogging, and try to prevent severe dry-to-wet swings that can affect crop development.",
        "soil": "Loose, well-drained soil supports root growth and tuber development. Prepare the bed without excessive compaction and follow local soil-test recommendations. Earthing-up practices and planting depth should follow regional crop guidance.",
        "routine": "Inspect lower leaves, upper leaves and stems on a regular schedule. Note where symptoms begin and whether nearby plants show similar changes. Keep records of planting date, irrigation and recent weather.",
        "monitor": "Watch for dark or expanding leaf lesions, yellowing, wilting and rapid changes after wet weather. Early and late blight can progress quickly under favourable conditions, so document suspected symptoms and seek local crop advice promptly.",
        "seasonal": "Scout more frequently during periods of humid or rainy weather. Maintain suitable soil moisture during tuber development and follow local recommendations for harvest timing and storage conditions.",
        "classes": "Early blight, late blight and healthy potato."
    },
    "Raspberry": {
        "light": "Raspberries generally benefit from a sunny position with good airflow. Provide a support system where needed and leave enough space between rows for access and air movement.",
        "water": "Keep the root zone supplied with moisture during establishment and fruiting. Shallow roots can dry during hot weather, so check soil moisture regularly. Avoid saturated soil and direct water toward the base where practical.",
        "soil": "Well-drained soil with organic matter is commonly preferred. Prepare the growing area before planting and check local soil conditions. Good drainage is especially important where rainfall is frequent.",
        "routine": "Use pruning guidance suited to the raspberry type, because summer-bearing and everbearing varieties are managed differently. Tie canes to supports where needed and remove damaged or spent canes at the appropriate time.",
        "monitor": "Inspect leaves and canes for unusual spots, discolouration, wilting or dieback. Check fruit for mould or damage and note whether problems are concentrated in dense, poorly ventilated areas.",
        "seasonal": "During flowering and fruiting, keep moisture consistent and harvest ripe fruit regularly. After the crop, identify which canes should be retained or removed based on the variety's fruiting habit.",
        "classes": "Healthy raspberry."
    },
    "Soybean": {
        "light": "Soybean requires an open, sunny field. Follow recommended plant population and row spacing for the variety and local production system so plants have suitable access to light and airflow.",
        "water": "Moisture requirements change throughout the crop cycle. Avoid prolonged drought during flowering and pod development, but also prevent waterlogging. Monitor different parts of the field because drainage and soil moisture may vary.",
        "soil": "Use locally recommended soil testing and fertility guidance. Well-drained soil supports root development, while compaction can restrict growth. Nutrient decisions should account for field history and local recommendations.",
        "routine": "Scout the crop systematically and record where symptoms first appear. Compare healthy and affected plants, inspect both sides of leaves and note growth stage, recent weather and field conditions.",
        "monitor": "Look for unusual yellowing, leaf spots, distortion, insect feeding or uneven plant growth. Check whether symptoms follow field edges, low-lying wet areas or particular rows; patterns can help guide further investigation.",
        "seasonal": "Increase field scouting during flowering and pod filling. After rainfall or other major weather events, revisit previously affected areas and record whether symptoms have changed.",
        "classes": "Healthy soybean."
    },
    "Squash": {
        "light": "Squash needs a sunny growing area and enough room for vines and broad leaves to spread. Use spacing suitable for the variety and growing method, including containers or trellises where appropriate.",
        "water": "Water deeply near the soil surface when needed, especially during flowering and fruit development. Avoid frequent shallow watering and try not to keep foliage wet for long periods. Good drainage helps prevent root stress.",
        "soil": "Fertile, well-drained soil supports vigorous growth. Add organic matter where appropriate and use soil-test or local crop guidance before adding nutrients.",
        "routine": "Check upper and lower leaf surfaces regularly. Guide vines gently if they are being trellised, and avoid damaging stems. Remove badly damaged material only when appropriate and keep the growing area reasonably clear.",
        "monitor": "Look for pale powder-like patches, yellowing, leaf distortion and fruit damage. Powdery mildew can spread across foliage, especially when plants are crowded. Similar symptoms can also result from other stresses.",
        "seasonal": "Inspect leaves more frequently as the canopy expands and during humid periods. Maintain airflow and monitor fruit development. Harvest according to variety guidance rather than leaving over-mature fruit on the plant.",
        "classes": "Powdery mildew."
    },
    "Strawberry": {
        "light": "Strawberries generally benefit from a sunny site. Good light supports flowering and fruit development, while spacing helps leaves dry after rain. Protect plants from extreme heat where the variety or local conditions require it.",
        "water": "Keep the root zone evenly moist during flowering and fruiting, but avoid waterlogging. Water near the base when possible so fruit and foliage do not remain wet for long periods. Containers may dry more quickly than garden beds.",
        "soil": "Use well-drained soil with organic matter. The crown should sit at the correct height: avoid burying it deeply or leaving roots exposed. Raised beds or suitable containers can help where garden drainage is poor.",
        "routine": "Remove dead or damaged leaves when appropriate and keep fruit from resting on wet soil. Check runners and manage them according to whether the goal is fruit production or propagation. Inspect fruit regularly during harvest.",
        "monitor": "Look for scorched or spotted leaves, mouldy fruit, weak growth and unusual crown or root symptoms. Note whether damage appears after hot weather, prolonged rain or changes in watering.",
        "seasonal": "During flowering and fruiting, maintain steady moisture and harvest ripe fruit frequently. After the main harvest, review plant health and follow variety-specific guidance for renovation and runner management.",
        "classes": "Leaf scorch and healthy strawberry."
    },
    "Tomato": {
        "light": "Tomatoes need a sunny, warm growing location. Provide enough space for the variety's mature size and use stakes or cages where appropriate. In very hot conditions, excessive heat can affect flowering and fruit set.",
        "water": "Water consistently at the root zone. The amount and frequency depend on container size, soil, weather and plant stage. Avoid repeatedly wetting foliage and try to prevent large swings between very dry and saturated soil.",
        "soil": "Use fertile, well-drained soil with suitable organic matter. Containers need adequate volume and drainage holes. Follow local soil-test and crop guidance for nutrients; too much fertiliser can create other problems.",
        "routine": "Inspect lower and upper leaves, stems and fruit at least weekly. Support plants as they grow, maintain airflow and remove fallen debris where appropriate. Clean tools between plants when disease is suspected and avoid handling wet foliage.",
        "monitor": "Look for circular or irregular leaf spots, yellowing, curling, pale patches, mould-like growth, wilting or fruit blemishes. Check whether symptoms start on lower leaves or new growth and whether they spread after humid or rainy weather. Several tomato diseases and non-disease stresses can look alike.",
        "seasonal": "During rapid growth, flowering and fruiting, monitor watering and inspect foliage regularly. After heavy rain or humid weather, check for new leaf spots. Keep notes about symptom progression and compare more than one leaf before drawing conclusions.",
        "classes": "Bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot, spider mites, target spot, mosaic virus, yellow leaf curl virus and healthy tomato."
    },
    "Rice": {
        "light": "Rice generally grows best in an open field with strong sunlight. Choose a variety suited to the local season, water availability and growing system. Good field levelling helps manage water consistently where flooded cultivation is practised.",
        "water": "Water management depends on the rice production system. Some systems use controlled shallow flooding, while others use saturated or intermittently irrigated soil. Not every rice crop needs continuous standing water; follow local agricultural recommendations and manage drainage carefully.",
        "soil": "Rice can be grown in different soil types, but water-holding capacity, drainage and soil fertility affect performance. Use soil testing and local recommendations to guide nutrient management. Field levelling and suitable drainage help maintain more even crop conditions.",
        "routine": "Inspect the crop regularly from seedling establishment through flowering and grain filling. Check plant density, weeds, water distribution and leaf colour. Scout several parts of the field rather than judging crop health from a single plant.",
        "monitor": "Look for leaf spots, lesions, unusual yellowing, leaf-tip drying, stem weakness, lodging or uneven patches. Record the crop stage and field location where symptoms appear. Nutrient stress, water stress, pests and diseases can produce similar-looking symptoms.",
        "seasonal": "During establishment, monitor germination and water distribution. During tillering and reproductive growth, check crop vigour and field conditions regularly. Near grain filling and harvest, watch for lodging, uneven maturity and changes after heavy rain.",
        "classes": "Rice is an additional educational care profile and is not one of the current 38 classes recognised by the PlantCare AI image classifier."
    }
}

CROP_CATEGORIES = {
    "Cereals and field crops": ["Corn", "Rice", "Soybean"],
    "Vegetables and garden crops": ["Pepper", "Potato", "Squash", "Tomato"],
    "Fruit trees and vines": ["Apple", "Cherry", "Grape", "Orange", "Peach"],
    "Berries": ["Blueberry", "Raspberry", "Strawberry"],
}



# ============================================================
# SESSION STATE
# ============================================================
def clear_image_results():
    for key in (
        "plant_predictions", "plant_image_hash", "gemini_answer",
        "gemini_answer_key", "plant_chat", "comparison_results"
    ):
        st.session_state.pop(key, None)

PAGES = ["Home", "Diagnose a Leaf", "Compare Models", "Plant Care Guide", "Crop Care Profiles", "About"]
if "page_selection" not in st.session_state:
    st.session_state["page_selection"] = "Home"


def navigate_to_page(target_page):
    """Set the sidebar navigation value from a Streamlit button callback."""
    st.session_state["page_selection"] = target_page

with st.sidebar:
    st.markdown('<div class="brand">🌿 PlantCare AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Plant health, with a little more clarity.</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="color:#D8E2D5;font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;">Explore</div>', unsafe_allow_html=True)
    st.radio("Navigate", PAGES, key="page_selection", label_visibility="collapsed")
    st.divider()
    status = model_file_status()
    st.markdown("**MODEL STATUS**")
    st.caption(("🟢" if status["Custom CNN"] else "🔴") + " Custom CNN")
    st.caption(("🟢" if status["MobileNetV2"] else "🔴") + " MobileNetV2")

page = st.session_state["page_selection"]
labels = None
try:
    if CLASS_PATH.exists():
        labels = load_class_names(str(CLASS_PATH))
except Exception as exc:
    st.sidebar.error(f"Label map issue: {exc}")

# ============================================================
# HOME
# ============================================================
if page == "Home":
    st.markdown("""
    <div class="hero">
      <div style="font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#D8E2D5;">🌱 PLANT HEALTH, MADE SIMPLE</div>
      <h1>A healthier harvest<br>starts with a leaf.</h1>
      <p>PlantCare AI explores how deep learning can classify plant-leaf images across 38 healthy and disease categories, with practical information to help users understand the results.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="kicker">EXPLORE PLANTCARE AI</div>', unsafe_allow_html=True)
    st.header("From leaf image to useful insight.")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            '<div class="soft-card home-feature-card">'
            '<div class="feature-icon">📷</div>'
            '<h3>Leaf diagnosis</h3>'
            '<p>Upload a leaf image and view a prediction from either trained model.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.button(
            "Analyse a leaf →",
            key="home_diagnose",
            type="primary",
            use_container_width=True,
            on_click=navigate_to_page,
            args=("Diagnose a Leaf",),
        )

    with c2:
        st.markdown(
            '<div class="soft-card home-feature-card">'
            '<div class="feature-icon">⚖️</div>'
            '<h3>Model comparison</h3>'
            '<p>Compare predictions from the Custom CNN and MobileNetV2.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.button(
            "Compare models →",
            key="home_compare",
            use_container_width=True,
            on_click=navigate_to_page,
            args=("Compare Models",),
        )

    with c3:
        st.markdown(
            '<div class="soft-card home-feature-card">'
            '<div class="feature-icon">🌿</div>'
            '<h3>Plant care library</h3>'
            '<p>Explore watering, light, soil, leaf symptoms and routine checks.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.button(
            "Open plant guide →",
            key="home_guide",
            use_container_width=True,
            on_click=navigate_to_page,
            args=("Plant Care Guide",),
        )

# ============================================================
# DIAGNOSE
# ============================================================
elif page == "Diagnose a Leaf":
    st.markdown('<div class="kicker">LEAF EXPLORER</div>', unsafe_allow_html=True)
    st.title("Let's take a closer look.")
    st.markdown('<p style="color:#65766B;">Upload a clear leaf photo, choose a model and explore its prediction.</p>', unsafe_allow_html=True)

    if labels is None:
        st.error("The 38-class class_names.json file is missing or invalid. Place it inside the models folder.")
    else:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            uploaded = st.file_uploader(
                "Choose a plant image",
                type=["jpg", "jpeg", "png"],
                key="leaf_upload_final",
                help="For best results, use a clear image with one leaf in focus.",
            )
            image = None
            image_hash = None
            if uploaded is not None:
                try:
                    raw_bytes = uploaded.getvalue()
                    image_hash = hashlib.sha256(raw_bytes).hexdigest()
                    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                    if st.session_state.get("plant_image_hash") != image_hash:
                        clear_image_results()
                        st.session_state["plant_image_hash"] = image_hash
                except Exception:
                    st.error("This image could not be opened. Try another JPG or PNG.")
            else:
                clear_image_results()

            available_models = [name for name, exists in model_file_status().items() if exists]
            if not available_models:
                st.error("No trained model files were found in the models folder.")
                selected_model = None
            else:
                selected_model = st.selectbox(
                    "Select model",
                    available_models,
                    key="selected_model_final",
                )
            analyse = st.button(
                "🌿 Analyse image",
                type="primary",
                use_container_width=True,
                disabled=(image is None or selected_model is None),
            )
            if st.button("↻ Clear image and start again", use_container_width=True):
                clear_image_results()
                st.session_state.pop("plant_image_hash", None)
                st.session_state.pop("leaf_upload_final", None)
                st.rerun()

        with col2:
            st.markdown("#### Image preview")
            if image is None:
                st.markdown('<div class="empty-preview"><div style="font-size:2.2rem;">🌿</div><b>Your leaf will appear here</b><p>Upload an image to begin.</p></div>', unsafe_allow_html=True)
            else:
                preview = Image.new("RGB", (720, 450), "#E7E9DC")
                thumb = image.copy()
                thumb.thumbnail((690, 420))
                preview.paste(thumb, ((720-thumb.width)//2, (450-thumb.height)//2))
                st.image(preview, use_container_width=True)
                a, b = st.columns(2)
                a.metric("Width", f"{image.width}px")
                b.metric("Height", f"{image.height}px")
                st.caption(f"{uploaded.name} · {uploaded.size/1024:.1f} KB")

        if analyse and image is not None and selected_model:
            try:
                with st.spinner(f"{selected_model} is analysing the leaf..."):
                    prediction = predict_with_model(image, selected_model, labels)
                st.session_state["plant_predictions"] = {selected_model: prediction}
                st.session_state["plant_prediction_hash"] = image_hash
                st.session_state["plant_prediction_image_name"] = uploaded.name
                history = st.session_state.get("prediction_history", [])
                crop, disease = human_label(prediction[0][0])
                history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "model": selected_model,
                    "crop": crop,
                    "prediction": disease,
                    "score": prediction[0][1],
                })
                st.session_state["prediction_history"] = history[:10]
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        predictions = st.session_state.get("plant_predictions", {})
        if image_hash and st.session_state.get("plant_prediction_hash") == image_hash and predictions:
            model_used = list(predictions.keys())[0]
            top = predictions[model_used]
            raw_label, score = top[0]
            crop, disease = human_label(raw_label)
            st.divider()
            st.markdown('<div class="kicker">MODEL PREDICTION</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-banner"><div class="result-label">{model_used.upper()} RESULT</div>'
                f'<div class="result-name">{disease}</div><div class="result-copy">{crop}</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0), 1), text=f"Model confidence score · {score:.1%}")
            st.markdown("### 🌿 Plant Health Assistant")

            answer_key = f"{image_hash}:{model_used}:{raw_label}"
            if st.button("Generate AI explanation", type="primary", key="generate_ai_final"):
                try:
                    with st.spinner("Getting a quick explanation..."):
                        answer = generate_ai_explanation(raw_label, model_used)
                    st.session_state["gemini_answer"] = answer
                    st.session_state["gemini_answer_key"] = answer_key
                    st.session_state.pop("gemini_error", None)
                except Exception as exc:
                    st.session_state["gemini_error"] = str(exc)

            if st.session_state.get("gemini_answer_key") == answer_key and st.session_state.get("gemini_answer"):
                with st.container(border=True):
                    st.markdown(st.session_state["gemini_answer"])
                st.caption("AI-generated learning support based on the predicted class.")
            elif st.session_state.get("gemini_error"):
                st.warning(st.session_state["gemini_error"])

            sample_questions = [
                "What does this plant disease mean?",
                "What signs should I look for on the leaves?",
                "How can I check whether the problem is spreading?",
                "What basic care should I give this plant?",
                "Could something other than disease cause these symptoms?",
            ]
            with st.expander("Ask a follow-up question"):
                selected_question = st.selectbox(
                    "Choose a question",
                    sample_questions,
                    key="plant_followup_select",
                )
                if st.button("Ask Plant Health Assistant", key="ask_followup_final"):
                    try:
                        with st.spinner("Preparing a short answer..."):
                            followup = generate_followup_answer(raw_label, selected_question)
                        st.session_state["plant_followup_answer"] = followup
                        st.session_state["plant_followup_key"] = answer_key + ":" + selected_question
                    except Exception as exc:
                        st.warning(
                            "The assistant could not answer right now. Your prediction is still available. "
                            "Please try again shortly."
                        )
                followup_key = answer_key + ":" + selected_question
                if (st.session_state.get("plant_followup_key") == followup_key
                        and st.session_state.get("plant_followup_answer")):
                    st.markdown(st.session_state["plant_followup_answer"])


            st.markdown("### 📄 Download your prediction report")
            report_explanation = ""
            if (st.session_state.get("gemini_answer_key") == answer_key
                    and st.session_state.get("gemini_answer")):
                report_explanation = st.session_state["gemini_answer"]
            try:
                pdf_data = create_prediction_pdf(
                    leaf_image=image,
                    crop=crop,
                    disease=disease,
                    model_name=model_used,
                    confidence=score,
                    ai_explanation=report_explanation,
                )
                st.download_button(
                    "Download prediction report (PDF)",
                    data=pdf_data,
                    file_name=f"PlantCare_AI_{crop.replace(' ', '_')}_Report.pdf",
                    mime="application/pdf",
                    key="download_plantcare_report",
                    use_container_width=True,
                )
                st.caption("Includes the leaf image, prediction, model confidence and any generated AI explanation.")
            except ImportError:
                st.info("To enable PDF reports, install ReportLab: python -m pip install reportlab")
            except Exception as exc:
                st.warning(f"Could not prepare the PDF report: {exc}")

# ============================================================
# MODEL COMPARISON
# ============================================================
elif page == "Compare Models":
    st.markdown('<div class="kicker">MODEL LAB</div>', unsafe_allow_html=True)
    st.title("Two models. One classification task.")
    st.markdown("Compare the Custom CNN and MobileNetV2 using the same uploaded image.")
    if labels is None:
        st.error("The class mapping is unavailable.")
    else:
        comparison_image = st.file_uploader(
            "Upload a leaf image to compare both models",
            type=["jpg", "jpeg", "png"],
            key="comparison_upload",
        )
        if comparison_image is not None:
            img = ImageOps.exif_transpose(Image.open(comparison_image)).convert("RGB")
            st.image(img, caption="Image used for both model predictions", width=360)
            if st.button("⚖️ Compare both models", type="primary"):
                results = {}
                for model_name in ("Custom CNN", "MobileNetV2"):
                    if not model_file_status()[model_name]:
                        st.warning(f"{model_name} file is missing.")
                        continue
                    try:
                        with st.spinner(f"Running {model_name}..."):
                            results[model_name] = predict_with_model(img, model_name, labels)
                    except Exception as exc:
                        st.error(f"{model_name}: {exc}")
                st.session_state["comparison_results"] = results
                st.session_state["comparison_image_hash"] = hashlib.sha256(comparison_image.getvalue()).hexdigest()

        results = st.session_state.get("comparison_results", {})
        if results:
            st.markdown("### Predictions for this image")
            cols = st.columns(len(results))
            for col, (model_name, ranked) in zip(cols, results.items()):
                crop, disease = human_label(ranked[0][0])
                with col:
                    st.markdown(f'<div class="soft-card"><h3>{model_name}</h3><h2>{disease}</h2><p>{crop}</p><b>Confidence: {ranked[0][1]:.1%}</b></div>', unsafe_allow_html=True)
            if len(results) == 2:
                same = results["Custom CNN"][0][0] == results["MobileNetV2"][0][0]
                st.info("Both models predicted the same class." if same else "The models predicted different classes. This disagreement is a useful reason to inspect the image and test labels; it does not establish which model is correct.")

    st.markdown("### Reported model evaluation")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="soft-card"><h3>Custom CNN</h3><h2>92.40%</h2><p>Reported test accuracy</p><p>Test loss: 0.2233</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="soft-card"><h3>MobileNetV2 · Transfer Learning</h3><h2>96.11%</h2><p>Test accuracy</p><p>Macro precision: 95.61%</p><p>Macro recall: 94.78%</p><p>Macro F1-score: 94.99%</p><p>Test loss: 0.1190</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><strong>Reading the results:</strong> MobileNetV2 has the higher reported test accuracy in the supplied project results.</div>', unsafe_allow_html=True)

# ============================================================
# PLANT CARE GUIDE
# ============================================================
elif page == "Plant Care Guide":
    st.markdown('<div class="kicker">THE GROWING GUIDE</div>', unsafe_allow_html=True)
    st.title("A healthier plant starts with good habits.")
    st.markdown(
        '<p style="color:#65766B;">A practical guide to everyday plant care, from watering and soil to spotting early signs of stress.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("### The essentials")
    guide_cards = [
        ("💧", "Water the root zone",
         "Check the soil before watering. For many potted plants, water when the upper layer of mix has begun to dry, then let excess water drain away. Outdoor crops may need more frequent checks in hot, windy weather. Avoid keeping roots constantly waterlogged."),
        ("☀️", "Match light to the plant",
         "Different crops need different amounts of sun. Check the growing requirements for your plant and watch how it responds. Pale, stretched growth can indicate insufficient light; scorched or bleached patches may indicate excessive direct sun or heat."),
        ("🪴", "Start with healthy soil and drainage",
         "Use a growing medium suited to the plant. Containers should have drainage holes, and garden beds should not remain waterlogged. Avoid compacting soil around roots; healthy soil structure helps air and water reach them."),
        ("🌱", "Feed thoughtfully",
         "Plants need nutrients, but more fertiliser is not always better. Follow the product label and crop-specific guidance. Excess fertiliser can damage roots or encourage leafy growth at the expense of flowers and fruit. Do not fertilise a severely stressed plant without identifying the cause."),
        ("✂️", "Prune with clean tools",
         "Remove dead or badly damaged parts when appropriate for the plant. Use clean, sharp tools and clean them between plants when disease is suspected. Avoid removing large amounts of healthy foliage at once."),
        ("🌬️", "Give plants room to breathe",
         "Follow recommended spacing for the crop. Good spacing helps light reach leaves and allows air to move through the canopy. When watering, avoid unnecessarily wetting foliage, especially late in the day."),
    ]
    for icon, title, description in guide_cards:
        st.markdown(
            f'<div class="soft-card"><div class="feature-icon">{icon}</div>'
            f'<h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Learn to read your plant")
    symptom_data = [
        ("Yellowing leaves", "Possible causes include natural ageing, watering stress, nutrient issues, root problems or pests. Note whether older or newer leaves are affected and check soil moisture."),
        ("Brown or crispy edges", "Can be associated with heat, irregular watering, dry air, salt build-up or excess fertiliser. Review recent changes in care and weather."),
        ("Spots or patches", "May have fungal, bacterial, insect-related or non-infectious causes. Note their colour, shape, location and whether they spread; a single photo cannot confirm the cause."),
        ("Wilting", "Both dry soil and waterlogged roots can cause wilting. Check moisture and drainage before changing the watering routine."),
        ("Holes or chewed edges", "Inspect leaf undersides and nearby stems for insects or feeding signs. Identify the likely pest before considering control options."),
    ]
    for title, body in symptom_data:
        with st.expander(title):
            st.write(body)

    st.markdown("### A simple  weekly  plant  check")
    for item in [
        "Inspect the whole plant, including the undersides of leaves.",
        "Check soil moisture and container drainage.",
        "Look for new spots, discolouration, insects or unusual growth.",
        "Remove fallen plant debris where appropriate.",
        "Record changes in weather, watering, feeding or plant location.",
    ]:
        st.markdown(f"- {item}")

    st.markdown("### Common care mistakes to avoid")
    mistakes = [
        ("Watering by the calendar", "Plant water needs change with pot size, soil, weather, season and growth stage. Check the soil and plant instead of following a rigid schedule."),
        ("Ignoring drainage", "A pot without drainage can leave roots sitting in water. Choose containers and growing media that suit the plant."),
        ("Changing many things at once", "If a plant looks stressed, change one care factor at a time where possible and observe its response."),
        ("Treating before identifying the cause", "Similar symptoms can come from different problems. Identify the likely cause and follow reliable local crop guidance before applying treatments."),
    ]
    for title, description in mistakes:
        st.markdown(
            f'<div class="soft-card"><h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="callout"><strong>Good practice:</strong> Plant-care requirements depend on variety, growth stage, climate and growing method. Use crop-specific local guidance for consequential decisions.</div>', unsafe_allow_html=True)

# ============================================================
# CROP CARE PROFILES
# ============================================================
elif page == "Crop Care Profiles":
    st.markdown('<div class="kicker">CROP LIBRARY</div>', unsafe_allow_html=True)
    st.title("Care guidance, crop by crop.")
    st.markdown(
        "Explore practical growing guidance grouped by crop type. Rice is included as an additional care guide, but is not currently one of the model's 38 prediction classes."
    )

    st.markdown("### Browse by crop group")
    selected_category = st.radio(
        "Choose a category",
        list(CROP_CATEGORIES.keys()),
        key="crop_care_category",
        horizontal=True,
        label_visibility="collapsed",
    )
    crop_names = CROP_CATEGORIES[selected_category]
    selected_crop = st.selectbox(
        "Choose a crop",
        crop_names,
        key="crop_care_profile",
    )
    profile = CROP_CARE_PROFILES[selected_crop]
    st.caption(
        f"{len(CROP_CATEGORIES[selected_category])} crops in this group · "
        f"{len(CROP_CARE_PROFILES)} crop profiles in total"
    )

    st.markdown(
        f'<div class="hero"><div style="font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#D8E2D5;">CROP CARE PROFILE</div>'
        f'<h1>{selected_crop}</h1><p>A practical overview of growing conditions, routine care and plant-health observations.</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Growing conditions")
    care_columns = st.columns(2)
    care_items = [
        ("☀️ Light and location", profile["light"]),
        ("💧 Watering", profile["water"]),
        ("🪴 Soil and drainage", profile["soil"]),
        ("🌱 Routine care", profile["routine"]),
    ]
    for index, (heading, detail) in enumerate(care_items):
        with care_columns[index % 2]:
            st.markdown(
                f'<div class="soft-card" style="min-height:205px;"><h3>{heading}</h3><p>{detail}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown("### What to monitor")
    st.markdown(
        f'<div class="soft-card"><h3>🔎 Signs and changes to watch</h3><p>{profile["monitor"]}</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Seasonal care notes")
    st.markdown(
        f'<div class="soft-card"><h3>🗓️ During the growing cycle</h3><p>{profile["seasonal"]}</p></div>',
        unsafe_allow_html=True
    )

# ============================================================
# ABOUT
# ============================================================
elif page == "About":
    st.markdown('<div class="kicker">ABOUT THE PROJECT</div>', unsafe_allow_html=True)
    st.title("Technology rooted in care.")
    st.markdown("""
    <div class="hero">
      <h1>Meet PlantCare AI.</h1>
      <p>A student-built plant-leaf image classification project exploring two deep-learning approaches and a simple, accessible web interface.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The project at a glance")
    a, b, c, d = st.columns(4)
    a.metric("Plant classes", "38")
    b.metric("Dataset images", "54,303")
    c.metric("Training images", "38,012")
    d.metric("Models", "2")

    st.markdown("### Two approaches to image classification")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="soft-card"><h3>Custom CNN</h3><p>A convolutional neural network trained from scratch on the PlantVillage class set. Its supplied evaluation summary reports 92.40% test accuracy on 8,145 held-out images.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card"><h3>MobileNetV2</h3><p>A transfer-learning model adapted to 38 output classes. The supplied evaluation reports 96.11% test accuracy, 95.61% macro precision, 94.78% macro recall and 94.99% macro F1-score.</p></div>', unsafe_allow_html=True)

    st.markdown("### How PlantCare AI works")
    st.markdown("""
    1. **Image input:** the user uploads a JPG, JPEG or PNG leaf photograph.
    2. **Preprocessing:** the image is converted to RGB and resized to the model's input dimensions. Each model receives its own expected preprocessing.
    3. **Model inference:** the selected classifier produces scores for 38 plant-health classes.
    4. **Result display:** the app presents the predicted class, confidence score and optional top-three alternatives.
    5. **AI explanation:** Gemini can provide a plain-language educational explanation based on the predicted label.
    6. **Downloadable report:** users can save a PDF summary of the prediction and any generated explanation.
    """)

    st.markdown("### Project team")
    st.markdown(
        '<div class="soft-card">'
        '<p><strong>Prayanshi Garg</strong> — Development</p>'
        '<p><strong>Samriddhi Dhyani</strong> — Custom CNN</p>'
        '<p><strong>Vaidehi Agarwal</strong> — MobileNetV2</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Acknowledgements")
    st.markdown(
        '<div class="soft-card">'
        '<p>We thank <strong>Mrs. Kamini Bhatt</strong>, our project mentor, '
        'and <strong>Mr. Nishant Singh Bisht</strong>, our faculty teacher, '
        'for their guidance and support. We also acknowledge the support of '
        'the department and extend our gratitude to everyone for encouragement and learning.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Technology stack")
    st.markdown("- Python and Streamlit for the application\n- TensorFlow / Keras for deep-learning inference\n- Custom CNN and MobileNetV2 for classification\n- Pillow and NumPy for image handling\n- Google Gemini API for optional natural-language explanations")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">🌿 PlantCare AI · Growing knowledge, one leaf at a time<br>Custom CNN + MobileNetV2</div>',
    unsafe_allow_html=True,
)
