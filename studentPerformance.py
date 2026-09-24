import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Force a GUI backend so plots open in a visible window.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


DATASET_PATH = Path("/Users/manyaruhela/Desktop/student_performance_prediction.csv")
PLOTS_FOLDER = Path.cwd() / "study_tracker_outputs"


class SmartStudentStudyTracker:
    def __init__(self):
        self.study_records = []
        self.focus_start_time = None
        self.best_session_hours = 0.0
        self.model = None
        self.cleaned_df = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

        self.feature_columns = [
            "study_hours_per_week",
            "attendance_rate",
            "previous_grades",
            "participation_in_extracurricular_activities",
            "parent_education_level",
        ]
        self.target_column = "final_performance"

        PLOTS_FOLDER.mkdir(exist_ok=True)

    def print_header(self):
        print("\n" + "=" * 75)
        print("SMART STUDENT STUDY TRACKER AND PERFORMANCE PREDICTION SYSTEM")
        print("=" * 75)

    def print_menu(self):
        print("\nChoose an option:")
        print("1. Start focus mode study session")
        print("2. Add study record manually")
        print("3. View collected study records")
        print("4. Run full workflow (cleaning, training, graphs)")
        print("5. Predict final score for new student input")
        print("6. Exit")

    def get_float_input(self, prompt, minimum=0.0, maximum=None):
        while True:
            user_input = input(prompt).strip()
            try:
                value = float(user_input)
                if value < minimum:
                    print(f"Please enter a value greater than or equal to {minimum}.")
                    continue
                if maximum is not None and value > maximum:
                    print(f"Please enter a value less than or equal to {maximum}.")
                    continue
                return value
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    def get_day_label(self):
        return f"Day {len(self.study_records) + 1}"

    def start_focus_mode(self):
        print("\nFOCUS MODE")
        print("Press Enter to start the study session.")
        input()

        self.focus_start_time = time.time()
        print("Study session started.")
        print("Press Enter again to stop the study session.")
        input()

        end_time = time.time()
        total_seconds = end_time - self.focus_start_time
        study_hours = total_seconds / 3600

        print(f"\nStudy session completed.")
        print(f"Total Study Time: {study_hours:.4f} hours")

        attendance = self.get_float_input("Enter attendance percentage for today: ", 0, 100)
        previous_score = self.get_float_input("Enter previous score: ", 0, 100)

        self.add_record(
            day=self.get_day_label(),
            study_hours=study_hours,
            attendance=attendance,
            previous_score=previous_score,
            source="Focus Mode",
        )

    def add_manual_record(self):
        print("\nADD MANUAL STUDY RECORD")
        study_hours = self.get_float_input("Enter study hours for the day: ", 0)
        attendance = self.get_float_input("Enter attendance percentage: ", 0, 100)
        previous_score = self.get_float_input("Enter previous score: ", 0, 100)

        self.add_record(
            day=self.get_day_label(),
            study_hours=study_hours,
            attendance=attendance,
            previous_score=previous_score,
            source="Manual Entry",
        )

    def add_record(self, day, study_hours, attendance, previous_score, source):
        record = {
            "Day": day,
            "Study Hours": round(study_hours, 4),
            "Attendance Percentage": round(attendance, 2),
            "Previous Score": round(previous_score, 2),
            "Entry Source": source,
        }
        self.study_records.append(record)

        print("\nStudy record added successfully.")
        self.check_milestone(study_hours)

    def check_milestone(self, study_hours):
        if study_hours > self.best_session_hours:
            self.best_session_hours = study_hours
            print("Milestone achieved!")
            print(f"Congratulations! New highest study session: {self.best_session_hours:.4f} hours")

    def view_study_records(self):
        if not self.study_records:
            print("\nNo study records available yet.")
            return

        df = pd.DataFrame(self.study_records)
        print("\nCOLLECTED STUDY RECORDS")
        print(df.to_string(index=False))

    def load_and_prepare_dataset(self):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

        df = pd.read_csv(DATASET_PATH)

        print("\nOriginal dataset shape:", df.shape)

        # Standardize column names.
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Remove duplicate rows.
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates().copy()
        print("Duplicate rows removed:", duplicates_before)

        # Convert numeric columns properly.
        numeric_columns = [
            "study_hours_per_week",
            "attendance_rate",
            "previous_grades",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean text columns.
        text_columns = [
            "participation_in_extracurricular_activities",
            "parent_education_level",
            "passed",
        ]
        for col in text_columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA})

        # Fill missing numeric values with median.
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())

        # Fill missing text values with mode.
        for col in text_columns:
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

        # Create a numeric final performance target for regression.
        participation_bonus = (
            df["participation_in_extracurricular_activities"]
            .str.lower()
            .map({"yes": 4, "no": 0})
            .fillna(0)
        )

        passed_bonus = (
            df["passed"]
            .str.lower()
            .map({"yes": 8, "no": -4})
            .fillna(0)
        )

        df[self.target_column] = (
            0.45 * df["previous_grades"]
            + 1.10 * df["study_hours_per_week"]
            + 0.25 * df["attendance_rate"]
            + participation_bonus
            + passed_bonus
        ).clip(0, 100)

        prepared_df = df[self.feature_columns + [self.target_column]].copy()
        self.cleaned_df = prepared_df

        print("Cleaned dataset shape:", prepared_df.shape)
        print("\nFirst five cleaned rows:")
        print(prepared_df.head().to_string(index=False))

        return prepared_df

    def train_model(self):
        df = self.load_and_prepare_dataset()

        X = df[self.feature_columns]
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        numeric_features = [
            "study_hours_per_week",
            "attendance_rate",
            "previous_grades",
        ]
        categorical_features = [
            "participation_in_extracurricular_activities",
            "parent_education_level",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median"))
                    ]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))
                    ]),
                    categorical_features,
                ),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\nMODEL TRAINING COMPLETED")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples : {len(X_test)}")
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R^2  : {r2:.4f}")

    def create_visualizations(self):
        if self.cleaned_df is None or self.model is None:
            self.train_model()

        # Use collected records if available, otherwise create sample daily records.
        if self.study_records:
            records_df = pd.DataFrame(self.study_records)
        else:
            records_df = pd.DataFrame([
                {"Day": "Day 1", "Study Hours": 2.5},
                {"Day": "Day 2", "Study Hours": 3.0},
                {"Day": "Day 3", "Study Hours": 4.2},
                {"Day": "Day 4", "Study Hours": 1.8},
                {"Day": "Day 5", "Study Hours": 3.6},
            ])
            print("\nNo manual study records found, so sample day-wise records were used for the bar graph.")

        # Plot 1: Days vs Study Hours
        plt.figure(figsize=(8, 5))
        plt.bar(records_df["Day"], records_df["Study Hours"], color="skyblue", edgecolor="black")
        plt.title("Days vs Study Hours", fontsize=14)
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("Study Hours", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_FOLDER / "days_vs_study_hours.png")
        plt.show(block=True)
        plt.close()

        # Plot 2: Study Hours vs Final Performance
        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.cleaned_df["study_hours_per_week"],
            self.cleaned_df[self.target_column],
            color="green",
            alpha=0.5,
            edgecolors="black",
        )
        plt.title("Study Hours vs Final Performance", fontsize=14)
        plt.xlabel("Study Hours per Week", fontsize=12)
        plt.ylabel("Final Performance", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_FOLDER / "study_hours_vs_final_performance.png")
        plt.show(block=True)
        plt.close()

        # Plot 3: Actual vs Predicted
        plt.figure(figsize=(8, 5))
        plt.scatter(self.y_test, self.y_pred, color="orange", alpha=0.6, edgecolors="black")
        min_value = min(self.y_test.min(), self.y_pred.min())
        max_value = max(self.y_test.max(), self.y_pred.max())
        plt.plot([min_value, max_value], [min_value, max_value], color="red", linewidth=2)
        plt.title("Actual vs Predicted Final Performance", fontsize=14)
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_FOLDER / "actual_vs_predicted.png")
        plt.show(block=True)
        plt.close()

        print("\nVISUALIZATIONS CREATED SUCCESSFULLY")
        print(f"Saved in folder: {PLOTS_FOLDER}")

    def predict_new_score(self):
        if self.model is None:
            print("\nModel is not trained yet. Training now...")
            self.train_model()

        print("\nPREDICT FINAL PERFORMANCE")
        study_hours = self.get_float_input("Enter study hours per week: ", 0)
        attendance = self.get_float_input("Enter attendance percentage: ", 0, 100)
        previous_score = self.get_float_input("Enter previous score: ", 0, 100)

        extracurricular = input(
            "Participation in extracurricular activities (Yes/No): "
        ).strip().title()
        if extracurricular not in {"Yes", "No"}:
            extracurricular = "No"

        parent_education = input(
            "Parent education level (High School/Associate/Bachelor/Master/PhD): "
        ).strip().title()
        if not parent_education:
            parent_education = "Unknown"

        new_data = pd.DataFrame([
            {
                "study_hours_per_week": study_hours,
                "attendance_rate": attendance,
                "previous_grades": previous_score,
                "participation_in_extracurricular_activities": extracurricular,
                "parent_education_level": parent_education,
            }
        ])

        prediction = float(self.model.predict(new_data)[0])
        prediction = max(0, min(100, prediction))

        print(f"\nPredicted Final Score: {prediction:.2f}")

    def run_full_workflow(self):
        self.train_model()
        self.create_visualizations()

    def run(self):
        self.print_header()
        print(f"Dataset path: {DATASET_PATH}")

        while True:
            self.print_menu()
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                self.start_focus_mode()
            elif choice == "2":
                self.add_manual_record()
            elif choice == "3":
                self.view_study_records()
            elif choice == "4":
                self.run_full_workflow()
            elif choice == "5":
                self.predict_new_score()
            elif choice == "6":
                print("\nThank you for using the system. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
    tracker = SmartStudentStudyTracker()
    tracker.run()
