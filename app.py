import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import process_grades, process_midterms
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit Code
st.set_page_config(layout="wide", page_title="🎯 GPA Predictor + Grade Analysis")
st.title("🎓 GPA Predictor + Required Grade Analyzer")
st.markdown("Upload predicted_semester_gpa.csv to begin.")

uploaded_file = st.file_uploader("📂 Upload predicted_semester_gpa.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ==============================
    # 1️⃣ Load Required Files (CLI)
    # ==============================
    print("🔄 Loading required files (CLI)...")

    @st.cache_resource
    def load_resources():
        models_cli = {
            "model": joblib.load("trained_models/elastic_net_model.pkl"),
            "Random Forest": joblib.load("trained_models/random_forest_model.pkl"),
            "Ridge Regression": joblib.load("trained_models/ridge_regression_model.pkl"),
            "Lasso Regression": joblib.load("trained_models/lasso_regression_model.pkl")
        }
        scaler_cli = joblib.load("scaler.pkl")
        target_scaler_cli = joblib.load("target_scaler.pkl")
        selected_features_cli = joblib.load("selected_features.pkl")
        encoding_dict_cli = joblib.load("encoding_dict.pkl")

        print("✅ Files loaded successfully (CLI)!")
        return models_cli, scaler_cli, target_scaler_cli, selected_features_cli, encoding_dict_cli

    models_cli, scaler_cli, target_scaler_cli, selected_features_cli, encoding_dict_cli = load_resources()

    # ==============================
    # 2️⃣ Load & Preprocess Unseen Data (CLI)
    # ==============================
    print("🔄 Loading unseen data (CLI)...")

    try:
        unseen_data_cli = df.copy()
        if "Predicted Semester GPA" in unseen_data_cli.columns:
            actual_gpa_cli = unseen_data_cli[["Predicted Semester GPA"]].copy()
        else:
            raise ValueError("❌ ERROR: 'Predicted Semester GPA' column is missing in unseen data (CLI)!")
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        st.error(f"❌ ERROR: {e}")
        st.stop()

    drop_cols = {
        "Registration Number", "Specialization", "Branch", "Last Semester",
        "Predicted GPA (Linear Regression)", "Predicted GPA (Ridge Regression)", "Predicted GPA (Lasso Regression)",
    }
    unseen_data_cli.drop(columns=drop_cols.intersection(unseen_data_cli.columns), inplace=True)

    # ==============================
    # 📌 Compute Missing Features (CLI)
    # ==============================
    print("🔄 Computing missing features (CLI)...")

    if "Grades (S, A, B, C, D, E, F, N)" in unseen_data_cli.columns:
        unseen_data_cli["Average Grade"], unseen_data_cli["Grade Std Dev"] = zip(
            *unseen_data_cli["Grades (S, A, B, C, D, E, F, N)"].apply(process_grades)
        )
        unseen_data_cli.drop(columns=["Grades (S, A, B, C, D, E, F, N)"], inplace=True)
    else:
        print("Error: 'Grades (S, A, B, C, D, E, F, N)' column not found (CLI)")
        st.error("Error: 'Grades (S, A, B, C, D, E, F, N)' column not found.")
        st.stop()

    if "Midterm Scores (out of 50)" in unseen_data_cli.columns:
        unseen_data_cli["Average Midterm Score"], unseen_data_cli["Weighted Midterm Score"] = zip(
            *unseen_data_cli.apply(
                lambda row: process_midterms(row["Midterm Scores (out of 50)"], actual_gpa_cli.iloc[row.name, 0]), axis=1
            )
        )
        unseen_data_cli.drop(columns=["Midterm Scores (out of 50)"], inplace=True)
    else:
        print("Error: 'Midterm Scores (out of 50)' column not found (CLI)")
        st.error("Error: 'Midterm Scores (out of 50)' column not found.")
        st.stop()

    print("✅ Missing features computed successfully (CLI)!")

    unseen_data_cli.drop(columns=["Predicted Semester GPA"], inplace=True)

    # ==============================
    # 3️⃣ Ensure Correct Features (Hard Matching) (CLI)
    # ==============================
    print("🔄 Ensuring unseen data has the correct features (CLI)...")

    expected_features_cli = [str(col).strip() for col in selected_features_cli]
    unseen_data_cli.columns = [str(col).strip() for col in unseen_data_cli.columns]

    missing_features_cli = [col for col in expected_features_cli if col not in unseen_data_cli.columns]
    extra_features_cli = [col for col in unseen_data_cli.columns if col not in expected_features_cli]

    for col in missing_features_cli:
        print(f"⚠️ WARNING: Missing feature '{col}' detected. Adding as NaN...")
        unseen_data_cli[col] = np.nan

    if extra_features_cli:
        print(f"⚠️ WARNING: Extra features detected and will be removed: {extra_features_cli}")
        unseen_data_cli.drop(columns=extra_features_cli, inplace=True)

    unseen_data_cli = unseen_data_cli[expected_features_cli]

    # Apply encoding to categorical columns
    for col, mapping in encoding_dict_cli.items():
        if col in unseen_data_cli.columns:
            unseen_data_cli[col] = unseen_data_cli[col].map(mapping).fillna(-1)

    unseen_data_cli = unseen_data_cli.astype(np.float64)

    print("✅ Unseen data preprocessing completed successfully (CLI)!")

    # ==============================
    # 4️⃣ Optimize Ridge & Lasso with GridSearchCV (CLI)
    # ==============================
    print("🔄 Performing hyperparameter tuning for Ridge & Lasso (CLI)...")

    ridge_params = {"alpha": [0.1, 1, 10, 50, 100]}
    lasso_params = {"alpha": [0.001, 0.01, 0.1, 1, 10]}

    # Scale features before fitting
    scaled_unseen_data_cli = scaler_cli.transform(unseen_data_cli)

    # Apply same scaling to targets
    scaled_target_cli = target_scaler_cli.transform(actual_gpa_cli)

    ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="neg_mean_squared_error")
    lasso_cv = GridSearchCV(Lasso(), lasso_params, cv=5, scoring="neg_mean_squared_error")

    ridge_cv.fit(scaled_unseen_data_cli, scaled_target_cli.ravel())
    lasso_cv.fit(scaled_unseen_data_cli, scaled_target_cli.ravel())

    best_ridge = ridge_cv.best_estimator_
    best_lasso = lasso_cv.best_estimator_

    print(f"✅ Best Ridge Alpha: {ridge_cv.best_params_['alpha']}")
    print(f"✅ Best Lasso Alpha: {lasso_cv.best_params_['alpha']}")

    models_cli["Ridge Regression"] = best_ridge
    models_cli["Lasso Regression"] = best_lasso

    # ==============================
    # 5️⃣ Make Predictions with Each Model (CLI)
    # ==============================
    print("🔄 Making predictions (CLI)...")

    # Reorder and scale features
    unseen_data_cli = unseen_data_cli[selected_features_cli]
    scaled_unseen_data_cli = scaler_cli.transform(unseen_data_cli)

    predictions_cli = {}
    for model_name, model in models_cli.items():
        print(f"🔹 Predicting with {model_name}...")
        try:
            predictions_cli[model_name] = model.predict(scaled_unseen_data_cli).flatten()
        except ValueError as e:
            print(f"ValueError during prediction with {model_name}: {e}")
            print(unseen_data_cli.isna().sum())
            st.error(f"ValueError during prediction with {model_name}: {e}")
            st.stop()

    # Inverse transform predictions
    for model_name, pred in predictions_cli.items():
        predictions_cli[model_name] = target_scaler_cli.inverse_transform(pred.reshape(-1, 1)).flatten()

    # Save predictions
    for model_name, pred in predictions_cli.items():
        df[f"Predicted GPA ({model_name})"] = pred

    st.success("✅ Predictions made successfully!")
    st.dataframe(df.head())
    
    # Grade logic
    def get_grade_band(mean, std):
        return {
            'S': max(90, mean + 1.5 * std),
            'A': (mean + 0.5 * std, mean + 1.5 * std),
            'B': (mean - 0.5 * std, mean + 0.5 * std),
            'C': (mean - 1.0 * std, mean - 0.5 * std),
            'D': (mean - 1.5 * std, mean - 1.0 * std),
            'E': (mean - 2.0 * std, mean - 1.5 * std),
            'F': (0, mean - 2.0 * std)
        }

    def analyze_subject_targets(midterm_list, internals, attendance, subject_credits, mean_std_list):
        required_grades = []
        required_term_end = []

        for i in range(len(midterm_list)):
            credit = subject_credits[i]
            midterm = midterm_list[i]
            mean, std = mean_std_list[i]

            if attendance < 3.25:
                required_grades.append('N')
                required_term_end.append(None)
                continue

            midterm_component = (midterm / 50) * 30
            current_total = midterm_component + internals + attendance
            bands = get_grade_band(mean, std)

            for grade in ['S', 'A', 'B', 'C', 'D', 'E']:
                lower_bound = bands[grade][0] if isinstance(bands[grade], tuple) else bands[grade]
                required_total = max(lower_bound, current_total)
                term_end_required = (required_total - current_total) / 0.3

                if 0 <= term_end_required <= 100:
                    required_grades.append(grade)
                    required_term_end.append(round(term_end_required, 2))
                    break
            else:
                required_grades.append('F')
                required_term_end.append(100.0)

        return required_grades, required_term_end

    # GPA Progress chart
    def show_progress(predicted_gpa, target_gpa):
        st.markdown("### 🎯 GPA Progress")
        progress = min(predicted_gpa / target_gpa, 1.0)
        st.progress(progress)
        st.markdown(f"**Predicted GPA:** {predicted_gpa:.2f} / Target GPA: {target_gpa:.2f}")
        st.markdown("---")

    # Radar Chart
    def show_radar_chart(subjects, required_grades):
        grade_points = {'S': 10, 'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 0, 'N': 0}
        scores = [grade_points.get(g, 0) for g in required_grades]

        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=subjects + [subjects[0]],
            fill='toself',
            name='Required Grade Levels'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            title='📈 Required Grade Strength Per Subject'
        )

        return fig

    reg_no = st.text_input("Enter Registration Number:")

    if reg_no:
        student_data = df[df["Registration Number"] == reg_no]

        if not student_data.empty:
            subject_names = st.text_input("Enter subjects (comma separated)", placeholder="Maths,Physics,DSA").split(',')
            subject_credits = st.text_input("Enter corresponding subject credits (comma separated)", placeholder="4,3,4").split(',')

            if len(subject_names) == len(subject_credits) and len(subject_names) > 0:
                subject_credits = list(map(int, subject_credits))
                total_results = []

                for idx, row in student_data.iterrows():
                    midterm_raw = str(row['Midterm Scores (out of 50)'])
                    midterm_scores = list(map(int, midterm_raw.split(',')))
                    internals = 35
                    attendance = round((row["Attendance (%)"] / 100) * 5, 2)
                    target_gpa = row["Predicted GPA (Lasso Regression)"]
                    mean_std_list = [(72, 8)] * len(subject_credits)

                    grades, terms = analyze_subject_targets(midterm_scores[:len(subject_credits)], internals, attendance, subject_credits, mean_std_list)

                    total_results.append({
                        "Registration Number": row["Registration Number"],
                        "Target GPA (Lasso)": target_gpa,
                        "Subjects": subject_names,
                        "Required Grades": grades,
                        "Required Term-End Marks": terms
                    })

                output_df = pd.DataFrame(total_results)
                st.write("### 🎯 Required Grades per Subject")
                st.dataframe(output_df)

                csv = output_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Grade Requirements", csv, "required_grades.csv", "text/csv")

                st.subheader("📊 Personalized Performance Insights")
                for idx, result in output_df.iterrows():
                    st.markdown(f"### 🎓 Student: `{result['Registration Number']}`")
                    st.markdown(f"**🎯 Target GPA:** `{round(result['Target GPA (Lasso)'], 2)}`")

                    current_cgpa = df.loc[idx, "CGPA"] # Changed student_data to df
                    if result['Target GPA (Lasso)'] > current_cgpa:
                        st.success("📈 Positive Progress: GPA is higher than CGPA. Keep it up!")
                    elif result['Target GPA (Lasso)'] < current_cgpa:
                        st.warning("📉 Dip Detected: GPA is lower than CGPA. Push harder!")
                    else:
                        st.info("➖ Neutral Trend: GPA equals CGPA.")

                    high_risk_subjects = 0
                    for sub, grade, term in zip(result["Subjects"], result["Required Grades"], result["Required Term-End Marks"]):
                        if grade == 'N':
                            st.error(f"❌ {sub}: Low attendance. Cannot clear subject.")
                            continue
                        term = float(term)
                        if term >= 85:
                            st.warning(f"⚠️ {sub}: {term} marks required for {grade}. Prioritize!")
                            high_risk_subjects += 1
                        elif term >= 70:
                            st.info(f"🟡 {sub}: {term} marks required for {grade}. Manageable.")
                        else:
                            st.success(f"🟢 {sub}: {term} marks for {grade}. Low pressure.")

                    if high_risk_subjects >= len(subject_names) * 0.6:
                        st.error("🚨 Too many subjects need high grades. Strong push required.")

                    attendance = df.loc[idx, "Attendance (%)"] # Changed student_data to df
                    if attendance < 75:
                        st.warning(f"📉 Attendance below 75% ({attendance}%). Fix it!")

                    show_progress(result['Target GPA (Lasso)'], current_cgpa)
                    st.plotly_chart(show_radar_chart(result["Subjects"], result["Required Grades"]), key = f"radar_{idx}")
                    st.markdown("---")
            else:
                st.info("Please enter subjects and credits.")

        else:
            st.warning("Registration number not found.")
    else:
        st.info("Please upload predicted_semester_gpa.csv to get started!")
