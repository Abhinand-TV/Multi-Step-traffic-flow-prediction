import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import TrafficDataset
from model import TrafficTransformer
from config import Config
from llm import TrafficLLM
import random

import datetime




cfg = Config()
API_KEY = "AIzaSyBuLjvqmjwTI6ZJ0twJhyG_yQ7uHjDTdhs"

st.set_page_config(page_title="Traffic AI", layout="centered")
st.title("🚦 Smart Traffic Prediction System")
st.subheader("Select Prediction Time")

selected_time = st.time_input("Choose Time", datetime.time(10, 0))

locations = {
    0: "Downtown LA",
    1: "Hollywood",
    2: "Santa Monica",
    3: "Beverly Hills",
    4: "Long Beach"
}

selected_loc = st.selectbox(
    "Select Location",
    list(locations.values())
)

# get sensor index
loc_index = list(locations.keys())[list(locations.values()).index(selected_loc)]


@st.cache_resource
def load_all():

    dataset = TrafficDataset(
        cfg.DATA_PATH,
        cfg.SEQ_LEN,
        cfg.PRED_LEN,
        cfg.MAX_SAMPLES
    )

    input_dim = dataset.data.shape[1]

    model = TrafficTransformer(
     input_dim,
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        num_layers=cfg.NUM_LAYERS,
        pred_len=cfg.PRED_LEN
    )
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location="cpu"))
    model.eval()

    llm = TrafficLLM(API_KEY)

    return model, dataset, llm


model, dataset, llm = load_all()


def get_status(speeds):
    avg = np.mean(speeds)

    if avg < 20:
        return "Very High Traffic"
    elif avg < 35:
        return "High Traffic"
    elif avg < 55:
        return "Moderate Traffic"
    else:
        return "Low Traffic"
def get_severity(speeds):
    variance = np.var(speeds)

    if variance > 50:
        return "Unstable"
    return "Stable"

def get_trend(speeds):
    diff = speeds[-1] - speeds[0]

    if diff > 3:
        return "Improving"
    elif diff < -3:
        return "Worsening"
    return "Stable"
def time_to_index(time_obj, dataset_length):

    total_minutes = time_obj.hour * 60 + time_obj.minute
    day_fraction = total_minutes / (24 * 60)

    index = int(day_fraction * dataset_length)

    return index

def generate_simple_nlp(status, trend):
    if "High" in status:
        if trend == "Worsening":
            return "Traffic is high and getting worse. Avoid this route."
        elif trend == "Improving":
            return "Traffic is high but improving slowly. Conditions may get better."
        return "Traffic is high and stable. Expect delays."

    if "Moderate" in status:
        if trend == "Worsening":
            return "Traffic is moderate but building up. Expect delays soon."
        elif trend == "Improving":
            return "Traffic is easing and becoming smoother."
        return "Traffic is moderate and stable."

    if "Low" in status:
        if trend == "Worsening":
            return "Traffic is currently low but increasing. It may get busier soon."
        elif trend == "Improving":
            return "Traffic is low and improving. Very smooth conditions."
        return "Traffic is low and stable. Good time to travel."



predict_clicked = st.button("Predict Traffic")

if predict_clicked:

    idx = time_to_index(selected_time, len(dataset))
    
    idx = max(0, min(idx, len(dataset) - 1))

    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0)).numpy()[0]

    input_dim = dataset.data.shape[1]

    full_pred = np.zeros((len(pred), input_dim))
    full_pred[:, loc_index] = pred   # ✅ use selected location

    pred_real = dataset.scaler.inverse_transform(full_pred)[:, loc_index]

    status = get_status(pred_real)
    trend = get_trend(pred_real)

    # ✅ STORE EVERYTHING
    st.session_state.pred_real = pred_real
    st.session_state.status = status
    st.session_state.trend = trend

    st.session_state.pred_done = True

if "pred_done" in st.session_state and st.session_state.pred_done:

    pred_real = st.session_state.pred_real
    status = st.session_state.status
    trend = st.session_state.trend

    # 📊 Summary
    st.subheader("📊 Traffic Summary")

    avg_speed = round(np.mean(pred_real), 2)
    min_speed = round(np.min(pred_real), 2)
    max_speed = round(np.max(pred_real), 2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Speed", f"{avg_speed} km/h")
    with col2:
        st.metric("Peak Speed", f"{max_speed} km/h")
    with col3:
        st.metric("Lowest Speed", f"{min_speed} km/h")

    # 🚦 Status
    if "Low" in status:
        st.success(f"🚦 {status} | Trend: {trend}")
    elif "Moderate" in status:
        st.warning(f"🚦 {status} | Trend: {trend}")
    else:
        st.error(f"🚦 {status} | Trend: {trend}")

    st.divider()


    





    future_times = []

    base_time = datetime.datetime.combine(datetime.date.today(), selected_time)

    for i in range(len(pred_real)):
        future_time = base_time + datetime.timedelta(minutes=5*(i+1))
        future_times.append(future_time.strftime("%H:%M"))

    status = get_status(pred_real)
    trend = get_trend(pred_real)
    
    

   

    st.subheader("🕒 Upcoming Traffic")

    for t, s in list(zip(future_times, pred_real))[:3]:
        st.markdown(f"**{t}** → `{round(s,2)} km/h`")

    st.divider()

    st.subheader("🤖 AI Insight")
    try:
        report = llm.generate_report(pred_real.tolist(), status, trend)
    except Exception as e:
        print("LLM failed:", e)
        report = generate_simple_nlp(status, trend)
    st.info(report)



st.subheader("💬 Ask Traffic Assistant")

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# input
user_query = st.text_input(
    "e.g. Is it a good time to travel?",
    key="chat_input"
)

# ask button
ask_button = st.button("Ask")

# 🔥 Handle response
if ask_button and user_query and "pred_real" in st.session_state:

    speeds = st.session_state.pred_real
    status = st.session_state.status
    trend = st.session_state.trend

    query = user_query.lower()

# 🔥 OPTIONAL: detect tone
    if "regret" in query:
        tone = "emotional"
    elif "should" in query or "would" in query:
        tone = "advice"
    else:
        tone = "general"
    st.caption(f"🧠 Interpreted query: {query}")

    # 🔥 Smart responses
    if any(word in query for word in ["best", "fastest", "when"]):
        best_idx = np.argmax(speeds)
        best_time = future_times[best_idx]
        response = f"Best time to travel is around {best_time}."

    elif any(word in query for word in ["good", "go now", "leave", "travel now"]):
        if "Low" in status:
            response = "Yes, traffic is smooth. Good time to travel."
        elif "Moderate" in status:
            response = "Traffic is moderate. Some delays possible."
        else:
            response = "Traffic is heavy. Better to wait."

    elif any(word in query for word in ["delay", "late", "stuck"]):
        if "High" in status:
            response = "Yes, heavy delays expected."
        elif "Moderate" in status:
            response = "Some delays expected."
        else:
            response = "No major delays expected."
    elif any(word in query for word in ["cancel", "trip", "should i", "worth"]):
        if "Low" in status:
            response = "No, you don't need to cancel your trip. Traffic is smooth."
        elif "Moderate" in status:
            response = "You can go, but expect some delays."
        else:
            response = "It’s better to delay or reconsider your trip due to heavy traffic."

    elif any(word in query for word in ["why", "explain", "how"]):
        response = f"Traffic is {status.lower()} and {trend.lower()} based on predicted patterns."

    elif any(word in query for word in ["change", "trend", "improve", "worsen"]):
        if trend == "Improving":
            response = "Traffic is improving."
        elif trend == "Worsening":
            response = "Traffic is getting worse."
        else:
            response = "Traffic is stable."
    elif any(word in query for word in ["crowded", "busy"]):
            if "High" in status:
                response = "Yes, traffic is quite crowded right now."
            elif "Moderate" in status:
                response = "Traffic is somewhat busy."
            else:
                response = "No, traffic is not crowded. It's smooth."

    elif any(word in query for word in ["clear", "free", "empty"]):
            if "Low" in status:
                response = "Yes, the road is clear and traffic is smooth."
            elif "Moderate" in status:
                response = "The road is partially clear, but some traffic is present."
            else:
                response = "No, the road is not clear. Traffic is heavy."

    else:
        try:
            st.caption("🤖 Using LLM response")
            response = llm.generate_report(
                speeds.tolist(),
                status,
                trend,
                user_query
        )

        # 🔥 SAFETY: if LLM gives weak/generic response
            if "traffic is" in response.lower() and len(response) < 80:
                raise ValueError("Weak LLM response")

        except:
        # 🔥 SMART FALLBACK (decision-aware)
            if "Low" in status:
                response = "You won't regret going now. Traffic is smooth."
            elif "Moderate" in status:
                response = "You might face some delays, but it's manageable."
            else:
                response = "You may regret going now due to heavy traffic."
    # 🔥 ADD VARIATION HERE
    import random

# 🔥 Only add variation if NOT already similar
    if "delay" not in response.lower() and "traffic" not in response.lower():

        if "Low" in status:
            extras = [
                "You're good to go.",
                "Should be a smooth ride.",
                "No issues expected."
            ]
        
        elif "Moderate" in status:
            extras = [
                "Just plan a bit of buffer time.",
                "It might slow you down slightly."
            ]
        
        else:
            extras = [
                "Better to wait if possible.",
                "You may want to delay your trip."
            ]

        response = response + " " + random.choice(extras)
    # ✅ store chat
    st.session_state.last_user = user_query
    st.session_state.last_ai = response

  


# 🔥 Display chat history (ALWAYS OUTSIDE)
if "chat_history" in st.session_state:
    if "last_user" in st.session_state:
        st.markdown(f"**🧑 You:** {st.session_state.last_user}")

    if "last_ai" in st.session_state:
        st.markdown(f"**🤖 AI:** {st.session_state.last_ai}")

 

  
    