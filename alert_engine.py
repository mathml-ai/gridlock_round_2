import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai

# Make sure your API keys are set in your environment
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# =====================================================================
# STEP 1: DYNAMIC RISK FILTERING
# =====================================================================
def get_dynamic_alert_zones(csv_file='live_predictions.csv', max_k=15, min_violation_floor=10.0, dropoff_threshold=0.50):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Error: live_predictions.csv not found.")
        return None
        
    hotspots = df.sort_values(by="predicted_next_hour", ascending=False).reset_index(drop=True)
    peak_volume = hotspots.loc[0, "predicted_next_hour"]
    
    if peak_volume < min_violation_floor:
        print(f"City is quiet. Peak risk is only {peak_volume:.1f}. No alerts needed.")
        return pd.DataFrame() 
        
    dynamic_k = 0
    for rank in range(max_k):
        if rank >= len(hotspots): break
        current_volume = hotspots.loc[rank, "predicted_next_hour"]
        if current_volume < min_violation_floor: break
        if (current_volume / peak_volume) < dropoff_threshold: break
        dynamic_k += 1

    final_alert_df = hotspots.head(dynamic_k).copy()
    final_alert_df["risk_rank"] = range(1, len(final_alert_df) + 1)

    # Base patrol units on actual predicted volume
    def assign_units(volume):
        if volume >= 1000: return 3  
        elif volume >= 500: return 2 
        else: return 1               

    final_alert_df["recommended_units"] = final_alert_df["predicted_next_hour"].apply(assign_units)
    return final_alert_df


# =====================================================================
# STEP 2: GENERATE THE LLM ALERT
# =====================================================================
def generate_dispatch_alert(critical_df):
    if critical_df is None or critical_df.empty:
        return None

    data_string = ""
    for _, row in critical_df.iterrows():
        # FIX: Safely grab coordinates whether they are named 'hex_lat' or 'latitude'
        lat = row.get('hex_lat', row.get('latitude', 0.0))
        lon = row.get('hex_lon', row.get('longitude', 0.0))
        
        data_string += (
            f"\n"
            f"HOTSPOT RANK #{row['risk_rank']}\n"
            f"Latitude: {lat:.5f}\n"
            f"Longitude: {lon:.5f}\n"
            f"Forecasted Violations: {row['predicted_next_hour']:.0f}\n"
            f"Growth Rate: {row['growth_rate']:.2f}x\n"
            f"Main Road Block Events: {row['main_road_blocks']}\n"
            f"Neighbor Spillover Pressure: {row['neighbor_pressure']}\n"
            f"Recommended Patrol Units: {row['recommended_units']}\n"
            f"--------------------------------------------------\n"
        )

    try:
        with open("dispatch_prompt.txt", "r") as file:
            prompt_template = file.read()
    except FileNotFoundError:
        prompt_template = "You are an AI dispatcher. Summarize this traffic report and provide strategic dispatch orders:\n\n{live_data}"

    final_prompt = prompt_template.format(live_data=data_string)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(final_prompt)
    
    return response.text


# =====================================================================
# STEP 3: SEND THE EMAIL
# =====================================================================
def send_email_alert(alert_text, receiver_email):
    """Fires the generated LLM text via an automated email."""
    if not alert_text or not receiver_email:
        return
        
    print(f"Transmitting alert to central dispatch ({receiver_email})...")
    
    lines = alert_text.split('\n')
    subject = lines[0] if "Subject" in lines[0] else "🚨 URGENT: Ranked AI Dispatch Orders"
    body = '\n'.join(lines[1:]).strip()

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, receiver_email, text)
        server.quit()
        print(f"✅ Alert successfully emailed to {receiver_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# =====================================================================
# EXECUTION RUNNER
# =====================================================================
if __name__ == "__main__":
    print("\n--- INITIATING AI ALERT SYSTEM ---")

    # FIX: Updated to call the new dynamic function
    critical_zones = get_dynamic_alert_zones(csv_file="live_predictions.csv")

    final_alert_text = generate_dispatch_alert(critical_zones)

    if final_alert_text:
        print("\n=== GENERATED ALERT ===")
        print(final_alert_text)
        print("=======================\n")

        send_email_alert(
            final_alert_text,
            receiver_email="2022kucp1027@iiitkota.ac.in"
        )