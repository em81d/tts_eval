import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np



import os
from dotenv import load_dotenv
from hume import HumeClient
load_dotenv()
HUME_API_KEY = os.getenv("HUME_API_KEY")

from hume.expression_measurement.batch.types import Models, Prosody, InferenceBaseRequest


# sǝhan



#all of this is straight from python SDK from Hume
# local_filepaths = [
#     open("audios.zip", mode="rb"),
# ]
# configs = InferenceBaseRequest(
#     models=Models(
#         prosody=Prosody(granularity="utterance")
#     )
# )
# job_id = client.expression_measurement.batch.start_inference_job_from_local_file(
#     file=local_filepaths,
#     json=configs
# )
# print(f"Job ID: {job_id}")


import time
# while True:
#     job_details = client.expression_measurement.batch.get_job_details(id=job_id)
#     status = job_details.state.status
#     if status == "COMPLETED":
#         print("Job completed.")
#         break
#     elif status == "FAILED":
#         print("Job failed.")
#         break
#     print(f"Status: {status}")
#     time.sleep(3)

# predictions = client.expression_measurement.batch.get_job_predictions(id=job_id)
# for result in predictions:
#     source = result.source
#     print(f"\nSource: {source.filename}")
#     for file_prediction in result.results.predictions:
#         for group in file_prediction.models.prosody.grouped_predictions:
#             for prediction in group.predictions:
#                 print(f"\n  Text: {prediction.text}")
#                 top_emotions = sorted(prediction.emotions, key=lambda e: e.score, reverse=True)[:3]
#                 for emotion in top_emotions:
#                     print(f"    {emotion.name}: {emotion.score:.3f}")







#streamlit page



st.set_page_config(page_title="Hume AI Expression Lab", page_icon="🎭", layout="centered")

st.title("🎭 Expression Measurement Lab")
st.markdown("Upload an audio file to analyze emotional expressivity using Hume AI.")

# Initialize the Hume Client
if not HUME_API_KEY:
    st.error("Missing HUME_API_KEY in .env file!")
    st.stop()

client = HumeClient(api_key=HUME_API_KEY)


# The file uploader widget
uploaded_file = st.file_uploader("Upload audio (mp3, wav)", type=['mp4', 'mp3', 'flac','wav'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    # 1. Basic Metadata
    if st.button("Analyze Expression"):
        with st.status("Processing with Hume AI...", expanded=True) as status:
            try:
                # 2. Prepare the request
                st.write("📤 Uploading file to Hume...")
                configs = InferenceBaseRequest(
                    models=Models(prosody=Prosody(granularity="utterance"))
                )

                # We pass the uploaded_file directly in a list
                job_id = client.expression_measurement.batch.start_inference_job_from_local_file(
                    file=[uploaded_file], 
                    json=configs
                )
                
                st.write(f"✅ Job Started! (ID: `{job_id}`)")

                # 3. Polling for results
                while True:
                    job_status = client.expression_measurement.batch.get_job_details(id=job_id)
                    current_state = job_status.state.status
                    
                    if current_state == "COMPLETED":
                        status.update(label="Analysis Complete!", state="complete")
                        break
                    elif current_state == "FAILED":
                        status.update(label="Analysis Failed", state="error")
                        st.error("Hume job failed. Check file format.")
                        st.stop()
                    else:
                        st.write(f"⏳ Status: {current_state}...")
                        time.sleep(3) # Wait 3 seconds before checking again

                # 4. Display Results
                st.subheader("📊 Emotion Analysis Summary")
                predictions = client.expression_measurement.batch.get_job_predictions(job_id)
                # Hume returns a list of results (one per file)
                for result in predictions:
                    # Access the prosody model results
                    group_predictions = result.results.predictions[0].models.prosody.grouped_predictions
                    for group in group_predictions:
                        for idx, prediction in enumerate(group.predictions):
                            # Get timeframe of the speech
                            start = prediction.time.begin
                            end = prediction.time.end
                            
                            with st.expander(f"Utterance {idx+1} ({start:.1f}s - {end:.1f}s)"):
                                
                                #section 1 - percentages for top 3
                                with st.expander("🏆 Top 3 Emotions", expanded=False):
                                    # Sort all emotions by their score (highest first)
                                    sorted_emotions = sorted(
                                        prediction.emotions, 
                                        key=lambda x: x.score, 
                                        reverse=True
                                    )
                                    
                                    # Display Top 3
                                    top_3 = sorted_emotions[:3]
                                    cols = st.columns(3)
                                    for i, emotion in enumerate(top_3):
                                        cols[i].metric(
                                            label=emotion.name, 
                                            value=f"{emotion.score:.2%}"
                                        )

                                # section 2 - "peakiness" of audio - supposed to analyze expressiveness. I have issues with this I don't like it
                                with st.expander("📈 Expressivity Analysis"):
                                    # 1. Get all scores as a list
                                    all_scores = [e.score for e in prediction.emotions]
                                    # 2. Calculate "Peakiness" (The difference between the top emotion and the average)
                                    # This is a proxy for how much the voice "stands out" from a neutral baseline
                                    max_score = max(all_scores)
                                    avg_score = sum(all_scores) / len(all_scores)
                                    expressivity_score = max_score - avg_score

                                    # 3. Display the Metric
                                    st.write(f"**Expressivity Index:** {expressivity_score:.4f}")
                                    if expressivity_score > 0.5:
                                        st.success("High Expressivity Detected")
                                    elif expressivity_score > 0.2:
                                        st.warning("Moderate Expressivity")
                                    else:
                                        st.error("Low Expressivity / Monotone")

                                    # --- Visualization: Top 5 Emotions Bar Chart ---
                                    import pandas as pd
                                    top_10_df = pd.DataFrame([
                                        {"Emotion": e.name, "Score": e.score} for e in sorted_emotions[:10]
                                    ])
                                    st.bar_chart(top_10_df.set_index("Emotion"))

                                # section 3 - statistical variance 
                                with st.expander("📊 Statistical Variance (Research Data)"):
                                    # Calculate Standard Deviation using numpy
                                    std_dev = np.std(all_scores)
    
                                    col_std, col_desc = st.columns([1, 2])
                                    col_std.metric("Standard Deviation (σ)", f"{std_dev:.4f}")
    
                                 # Research Interpretation
                                    if std_dev > 0.1:
                                        col_desc.info("✨ **High Variance:** The speaker shows distinct emotional peaks.")
                                    elif std_dev > 0.05:
                                        col_desc.info("⚖️ **Moderate Variance:** Balanced emotional delivery.")
                                    else:
                                        col_desc.info("〰️ **Low Variance:** Likely a monotone or 'flat' delivery.")

                                    # Visualization: A histogram of all 48 emotion scores
                                    st.write("Distribution of Emotion Scores:")
                                    st.bar_chart(all_scores)
                                
                                with st.expander("📄 Segment JSON"):
                                    st.json(prediction.emotions)
                
                    # --- GLOBAL RESEARCH SUMMARY ---
                    st.divider()
                    st.header("🔬 Global Research Summary")
                    st.markdown("This section aggregates all data to provide a profile of the speaker's **overall** expressivity.")

                    # 1. Aggregate all emotion scores across all segments
                    all_segment_emotions = [] # To store a list of all emotion lists
                    for result in predictions:
                        group_predictions = result.results.predictions[0].models.prosody.grouped_predictions
                        for group in group_predictions:
                            for prediction in group.predictions:
                                # Create a dictionary of {emotion_name: score} for this segment
                                seg_dict = {e.name: e.score for e in prediction.emotions}
                                all_segment_emotions.append(seg_dict)

                    if all_segment_emotions:
                        # Convert to a DataFrame so we can easily average the columns
                        df_all = pd.DataFrame(all_segment_emotions)
                        
                        # Calculate the MEAN score for each of the 48 emotions across the whole file
                        global_means = df_all.mean()
                        
                        # Sort them for the Top 3
                        global_sorted = global_means.sort_values(ascending=False)

                        # --- SECTION A: Global Top 3 ---
                        with st.expander("🏆 Global Top 3 (Average)", expanded=True):
                            st.write("The most dominant emotions throughout the entire recording:")
                            cols = st.columns(3)
                            for i in range(3):
                                cols[i].metric(
                                    label=global_sorted.index[i], 
                                    value=f"{global_sorted.values[i]:.2%}"
                                )

                        # --- SECTION B: Global Expressivity Analysis ---
                        with st.expander("📈 Global Expressivity Analysis"):
                            # Peakiness for the whole file
                            global_max = global_sorted.max()
                            global_avg = global_sorted.mean()
                            global_peakiness = global_max - global_avg
                                
                            col_m, col_s = st.columns([1, 2])
                            col_m.write(f"**Avg Peakiness:** `{global_peakiness:.4f}`")
                            
                            if global_peakiness > 0.4:
                                col_s.success("Overall High Expressivity")
                            else:
                                col_s.info("Overall Consistent/Stable Delivery")

                            # Global Bar Graph (Top 10)
                            st.write("Average Intensity of Top 10 Emotions:")
                            st.bar_chart(global_sorted.head(10))

                        # --- SECTION C: Global Statistical Variance ---
                        with st.expander("📊 Global Statistical Variance"):
                            # We calculate the Std Dev of the averaged emotion profile
                            global_std = np.std(global_means.values)
                                
                            st.metric("Global Standard Deviation (σ)", f"{global_std:.4f}")
                            st.write("Distribution of Average Emotion Scores:")
                            
                            # This shows how "flat" or "peaky" the person's baseline is
                            st.bar_chart(global_means.values)
                            st.caption("A flatter graph here suggests a more neutral overall personality in this clip.")

                        # 4. Timeline Trend (Keeping the line chart from before)
                        st.write("### Expressivity Trend (Per Utterance)")
                        # Re-calculating peakiness per segment for the trend line
                        peak_trend = [max(row) - (sum(row)/len(row)) for row in df_all.values]
                        st.line_chart(peak_trend)

                    # 6. Data Export for Statistics
                    # summary_data = pd.DataFrame({
                    #     "Utterance": range(1, len(all_peakiness_scores) + 1),
                    #     "Peakiness": all_peakiness_scores,
                    #     "Std_Dev": all_std_devs
                    # })
                    
                    # st.download_button(
                    #     label="📥 Download Research Data (CSV)",
                    #     data=summary_data.to_csv(index=False),
                    #     file_name="expressivity_research_summary.csv",
                    #     mime="text/csv"
                    # )

                with st.expander("📂 Full Job JSON Response"):
                    results = client.expression_measurement.batch.get_job_predictions(job_id)
                    st.json(results) # Displaying the raw JSON for now

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Click \"Analyze Expression\" to begin")

