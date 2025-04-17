import pandas as pd, streamlit as st, plotly.express as px, numpy as np, tensorflow as tf, requests
from prophet import Prophet
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pymongo import MongoClient
from twilio.rest import Client  # For SMS notifications

st.set_page_config(page_title="India Disaster Analysis", layout="wide")
india_geojson = (
    "https://gist.githubusercontent.com/jbrobst/"
    "56c13bbbf9d97d187fea01ca62ea5112/raw/"
    "e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
)
ZONE_COLORS = {
    "Zone V (Very High Risk)": "#ff0000",
    "Zone IV (High Risk)": "#ffa500",
    "Zone III (Moderate Risk)": "#ffff00",
    "Zone II (Low Risk)": "#00ff00"
}

# --- Connect to MongoDB via Flask backend ---
@st.cache_data(show_spinner=True)
def get_df() -> pd.DataFrame:
    # Fetch data from the Flask backend endpoint
    url = "http://localhost:5000/data"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)
    st.write("Available columns in data:", df.columns.tolist())
    
    # Check for required columns; we expect 'Start Year', 'Start Month', and 'Location'
    required_cols = ['Start Year', 'Start Month', 'Location']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Please check your MongoDB collection schema.")
        return pd.DataFrame()
    
    # Fill missing 'Start Month' values and convert to datetime
    df['Start Month'] = df['Start Month'].fillna(1)
    df['Start Date'] = pd.to_datetime(
        df['Start Year'].astype(str) + '-' + df['Start Month'].astype(str) + '-01',
        errors='coerce'
    )
    
    # Rename "Location" to "Admin Units" if needed
    if "Location" in df.columns:
        # If "Admin Units" exists, fill missing values with "Location", then drop "Location"
        if "Admin Units" in df.columns:
            df["Admin Units"] = df["Admin Units"].fillna(df["Location"])
            df.drop(columns=["Location"], inplace=True)
        else:
            df.rename(columns={"Location": "Admin Units"}, inplace=True)
    
    # Handle duplicate column names for "Admin Units"
    admin_units = df["Admin Units"]
    if isinstance(admin_units, pd.DataFrame):
        admin_units = admin_units.iloc[:, 0]
    df["Admin Units"] = admin_units.astype(str).str.title().replace({
        "Nct Of Delhi": "Delhi",
        "Pondicherry": "Puducherry",
        "Orissa": "Odisha"
    })
    
    return df

def make_predictions(df):
    try:
        df_ts = df.resample('Y', on='Start Date').size().reset_index(name='Count')
        df_ts.columns = ['ds', 'y']
        if len(df_ts) < 2:
            raise ValueError("Need at least 2 years of data for forecasting")
        model = Prophet()
        model.fit(df_ts)
        future = model.make_future_dataframe(periods=5, freq='Y')
        return model.predict(future)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

STATE_ZONE_MAPPING = {
    "Andaman & Nicobar Islands": "Zone V (Very High Risk)",
    "Arunachal Pradesh": "Zone V (Very High Risk)",
    "Assam": "Zone V (Very High Risk)",
    "Meghalaya": "Zone V (Very High Risk)",
    "Mizoram": "Zone V (Very High Risk)",
    "Nagaland": "Zone V (Very High Risk)",
    "Tripura": "Zone V (Very High Risk)",
    "Sikkim": "Zone IV (High Risk)",
    "Jammu & Kashmir": "Zone IV (High Risk)",
    "Ladakh": "Zone IV (High Risk)",
    "Manipur": "Zone IV (High Risk)",
    "Uttarakhand": "Zone IV (High Risk)",
    "Himachal Pradesh": "Zone IV (High Risk)",
    "Bihar": "Zone IV (High Risk)",
    "Gujarat": "Zone IV (High Risk)",
    "Delhi": "Zone IV (High Risk)",
    "West Bengal": "Zone III (Moderate Risk)",
    "Uttar Pradesh": "Zone III (Moderate Risk)",
    "Haryana": "Zone III (Moderate Risk)",
    "Punjab": "Zone III (Moderate Risk)",
    "Maharashtra": "Zone III (Moderate Risk)",
    "Jharkhand": "Zone III (Moderate Risk)",
    "Odisha": "Zone III (Moderate Risk)",
    "Andhra Pradesh": "Zone III (Moderate Risk)",
    "Tamil Nadu": "Zone III (Moderate Risk)",
    "Kerala": "Zone III (Moderate Risk)",
    "Goa": "Zone III (Moderate Risk)",
    "Dadra & Nagar Haveli & Daman & Diu": "Zone III (Moderate Risk)",
    "Lakshadweep": "Zone III (Moderate Risk)",
    "Rajasthan": "Zone II (Low Risk)",
    "Madhya Pradesh": "Zone II (Low Risk)",
    "Chhattisgarh": "Zone II (Low Risk)",
    "Telangana": "Zone II (Low Risk)",
    "Karnataka": "Zone II (Low Risk)",
    "Puducherry": "Zone II (Low Risk)"
}

def plot_disaster_zones(df):
    df_zone_map = pd.DataFrame(list(STATE_ZONE_MAPPING.items()),
                               columns=["Admin Units", "Risk Zone"])
    fig = px.choropleth_mapbox(
        df_zone_map,
        geojson=india_geojson,
        locations="Admin Units",
        featureidkey="properties.ST_NM",
        color="Risk Zone",
        color_discrete_map=ZONE_COLORS,
        mapbox_style="carto-positron",
        zoom=3.5,
        center={"lat": 22.9734, "lon": 78.6569},
        opacity=0.7,
        height=700
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox=dict(bearing=0, pitch=0)
    )
    return fig

# --- Twilio SMS Notification Function ---
def send_sms_notification(message, to_phone):
    account_sid = "ACf88bd40f21c370eb6d1ec21e2f130949"          
    auth_token = "8c64bff0b72a9f591f4ca37a4ea266ec"             
    from_phone = "+16084078167"   -l
    client = Client(account_sid, auth_token)
    sms = client.messages.create(
        body=message,
        from_=from_phone,
        to=to_phone
    )
    return sms.sid

@st.cache_resource(show_spinner=True)
def train_evaluate_models(df):
    possible_features = ['Magnitude', 'Total Deaths', 'No. Injured', 'Total Affected', 'Days Difference']
    features = [col for col in possible_features if col in df.columns]
    df_class = df.dropna(subset=features + ['Disaster Type'])
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_class['Disaster Type'])
    labels = le.classes_
    X_train, X_test, y_train, y_test = train_test_split(
        df_class[features], y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    results = {}
    # Decision Tree Model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=range(len(labels)))
    overall_dt_fig = px.imshow(
        pd.DataFrame(cm_dt, index=labels, columns=labels),
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Overall Confusion Matrix: Decision Tree"
    )
    per_disaster_dt = {}
    for i, disaster in enumerate(labels):
        y_test_bin = (y_test == i).astype(int)
        y_pred_bin = (y_pred_dt == i).astype(int)
        cm_bin = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
        df_cm_bin = pd.DataFrame(
            cm_bin,
            index=[f"Actual Not {disaster}", f"Actual {disaster}"],
            columns=[f"Pred Not {disaster}", f"Pred {disaster}"]
        )
        per_disaster_dt[disaster] = px.imshow(
            df_cm_bin,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title=f"2x2 Confusion Matrix: {disaster} (Decision Tree)"
        )
    accuracy_dt = (y_pred_dt == y_test).mean()
    fi_dt = pd.DataFrame({'Feature': features, 'Importance': dt_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    fi_dt_fig = px.bar(fi_dt, x='Feature', y='Importance', title="Feature Importance (Decision Tree)")
    results['decision_tree'] = {
        'overall_cm_fig': overall_dt_fig,
        'per_disaster_cm_figs': per_disaster_dt,
        'accuracy': accuracy_dt,
        'fi_fig': fi_dt_fig
    }
    
    # Neural Network Model (Improved)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    num_classes = len(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(32, activation='relu'), BatchNormalization(), Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = nn_model.fit(
        X_train_scaled,
        y_train,
        epochs=200,
        validation_split=0.2,
        callbacks=[es, lr_scheduler],
        class_weight=class_weights_dict,
        verbose=0
    )
    history_dict = history.history
    loss_df = pd.DataFrame({'Epoch': list(range(1, len(history_dict['loss']) + 1)),
                             'Training Loss': history_dict['loss'],
                             'Validation Loss': history_dict['val_loss']})
    loss_fig = px.line(loss_df, x='Epoch', y=['Training Loss', 'Validation Loss'],
                       title="Neural Network: Training vs. Validation Loss")
    acc_df = pd.DataFrame({'Epoch': list(range(1, len(history_dict['accuracy']) + 1)),
                            'Training Accuracy': history_dict['accuracy'],
                            'Validation Accuracy': history_dict['val_accuracy']})
    acc_fig = px.line(acc_df, x='Epoch', y=['Training Accuracy', 'Validation Accuracy'],
                      title="Neural Network: Training vs. Validation Accuracy")
    y_pred_nn_prob = nn_model.predict(X_test_scaled)
    y_pred_nn = y_pred_nn_prob.argmax(axis=1)
    accuracy_nn = (y_pred_nn == y_test).mean()
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    roc_curves, pr_curves = {}, {}
    for i, disaster in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_nn_prob[:, i])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_nn_prob[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_pred_nn_prob[:, i])
        roc_curves[disaster] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        pr_curves[disaster] = {'precision': precision, 'recall': recall, 'avg_precision': avg_precision}
    results['neural_network'] = {
        'accuracy': accuracy_nn,
        'loss_history_fig': loss_fig,
        'acc_history_fig': acc_fig,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves
    }
    return results, le

def main():
    st.title("🇮🇳 India Disaster Zone Analysis")
    st.markdown("### Natural Disaster Risk Zones (MSK Scale)")
    df = get_df()
    tabs = st.tabs([
        "Disaster Zones",
        "Forecast Analysis",
        "Human Impact",
        "Economic Impact",
        "Prediction Evaluation",
        "Real-Time Input"
    ])
    
    with tabs[0]:
        st.header("Regional Risk Zone Distribution")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(plot_disaster_zones(df), use_container_width=True)
        with col2:
            st.markdown("**Zone Classification**")
            for zone, color in ZONE_COLORS.items():
                st.markdown(f"<span style='color:{color}'>■</span> {zone}", unsafe_allow_html=True)
    
    with tabs[1]:
        st.header("5-Year Disaster Forecast")
        forecast = make_predictions(df)
        if forecast is not None:
            st.plotly_chart(px.line(
                forecast,
                x='ds',
                y='yhat',
                title="Disaster Frequency Projection"
            ), use_container_width=True)
    
    with tabs[2]:
        st.header("Human Impact Analysis")
        metrics = ["Total Deaths"]
        if "No. Injured" in df.columns:
            metrics.append("No. Injured")
        df_melt = df.melt(
            id_vars=["Disaster Type"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Count"
        )
        st.plotly_chart(px.bar(
            df_melt,
            x="Disaster Type",
            y="Count",
            color="Metric",
            barmode="group",
            title="Casualty Statistics by Disaster Type"
        ), use_container_width=True)
    
    with tabs[3]:
        st.header("Economic Impact Analysis")
        if "Total Damage ('000 US$)" in df.columns:
            st.plotly_chart(px.scatter(
                df,
                x='Magnitude',
                y="Total Damage ('000 US$)",
                color='Disaster Type',
                size='Total Affected',
                log_y=True,
                title="Damage Correlation Analysis"
            ), use_container_width=True)
        else:
            st.write("No economic damage data found.")
    
    with tabs[4]:
        st.header("Disaster Type Prediction Evaluation")
        results, le = train_evaluate_models(df)
        st.subheader("Decision Tree Model")
        st.write(f"**Accuracy:** {results['decision_tree']['accuracy']:.2f}")
        dt_option = st.selectbox(
            "Select Confusion Matrix to Display (Decision Tree)",
            options=["Overall"] + list(results['decision_tree']['per_disaster_cm_figs'].keys())
        )
        if dt_option == "Overall":
            st.plotly_chart(results['decision_tree']['overall_cm_fig'], use_container_width=True)
        else:
            st.plotly_chart(results['decision_tree']['per_disaster_cm_figs'][dt_option], use_container_width=True)
        st.plotly_chart(results['decision_tree']['fi_fig'], use_container_width=True)
        
        st.subheader("Neural Network Model")
        st.write(f"**Accuracy:** {results['neural_network']['accuracy']:.2f}")
        st.markdown("##### Training History")
        st.plotly_chart(results['neural_network']['loss_history_fig'], use_container_width=True)
        st.plotly_chart(results['neural_network']['acc_history_fig'], use_container_width=True)
        
        nn_class = st.selectbox(
            "Select Disaster Type for ROC & Precision-Recall Curves",
            options=list(results['neural_network']['roc_curves'].keys())
        )
        roc_data = results['neural_network']['roc_curves'][nn_class]
        pr_data = results['neural_network']['pr_curves'][nn_class]
        roc_df = pd.DataFrame({
            'False Positive Rate': roc_data['fpr'],
            'True Positive Rate': roc_data['tpr']
        })
        roc_fig = px.line(
            roc_df,
            x='False Positive Rate',
            y='True Positive Rate',
            title=f"ROC Curve for {nn_class} (AUC = {roc_data['auc']:.2f})"
        )
        roc_fig.add_shape(
            type='line',
            line=dict(dash='dash'),
            x0=0, y0=0, x1=1, y1=1
        )
        st.plotly_chart(roc_fig, use_container_width=True)
        pr_df = pd.DataFrame({
            'Recall': pr_data['recall'],
            'Precision': pr_data['precision']
        })
        pr_fig = px.line(
            pr_df,
            x='Recall',
            y='Precision',
            title=f"Precision-Recall Curve for {nn_class} (Avg Precision = {pr_data['avg_precision']:.2f})"
        )
        st.plotly_chart(pr_fig, use_container_width=True)
    
    with tabs[5]:
        st.header("Real-Time Input")
        region = st.selectbox("Select Your Region", list(STATE_ZONE_MAPPING.keys()))
        magnitude = st.number_input("Enter Current Magnitude", min_value=0.0, value=0.0, step=0.1)
        temperature = st.number_input("Enter Current Temperature (°C)", min_value=-50.0, value=25.0, step=0.1)
        
        # Text input for phone number in E.164 format for SMS notifications
        user_phone = st.text_input("Enter your phone number for SMS alerts (e.g., +1234567890)", value="")
        
        if st.button("Predict Disaster Situation"):
            zone = STATE_ZONE_MAPPING.get(region, "Unknown")
            if zone == "Zone V (Very High Risk)":
                risk = "Critical" if magnitude >= 7 else ("High" if magnitude >= 5 else "Moderate")
            elif zone == "Zone IV (High Risk)":
                risk = "High" if magnitude >= 6 else ("Moderate" if magnitude >= 4 else "Low")
            elif zone == "Zone III (Moderate Risk)":
                risk = "High" if magnitude >= 5 else ("Moderate" if magnitude >= 3 else "Low")
            elif zone == "Zone II (Low Risk)":
                risk = "Moderate" if magnitude >= 4 else "Low"
            else:
                risk = "Unknown"
            pred = ("Earthquake" if risk == "Critical" else
                    ("Flood" if risk == "High" else
                     ("Cyclone" if risk == "Moderate" else "Wind Hazard")))
            
            # Display on-screen details
            st.markdown(f"### Region: {region}")
            st.markdown(f"**Zone:** {zone}")
            st.markdown(f"**Current Magnitude:** {magnitude}")
            st.markdown(f"**Temperature:** {temperature} °C")
            st.markdown(f"**Predicted Risk Level:** {risk}")
            st.markdown(f"**Predicted Disaster Type:** {pred}")
            
            # Alert notification with detailed info
            risk_percentage_map = {
                "Critical": "90-100%",
                "High": "70-90%",
                "Moderate": "40-70%",
                "Low": "10-40%",
                "Unknown": "Not Available"
            }
            risk_percentage = risk_percentage_map.get(risk, "Not Available")
            alert_message = (
                f"Alert for {region}:\n"
                f"Zone: {zone}\n"
                f"Magnitude: {magnitude}\n"
                f"Risk Level: {risk} ({risk_percentage})\n"
                f"Predicted Disaster: {pred}"
            )
            st.info(alert_message)
            
            # If a phone number is provided, send SMS alert using Twilio
            if user_phone:
                # Twilio configuration: replace with your actual credentials
                account_sid = "ACf88bd40f21c370eb6d1ec21e2f130949"
                auth_token = "8c64bff0b72a9f591f4ca37a4ea266ec"
                from_phone = "+16084078167"
                from twilio.rest import Client
                client = Client(account_sid, auth_token)
                sms = client.messages.create(
                    body=alert_message,
                    from_=from_phone,
                    to=user_phone
                )
                st.success(f"SMS alert sent successfully. Message SID: {sms.sid}")

if __name__ == "__main__":
    main()
