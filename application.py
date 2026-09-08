from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, Predict_Pipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictions", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=request.form.get("reading_score"),
        writing_score=request.form.get("writing_score"),
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = Predict_Pipeline()
    explainability_result = predict_pipeline.explain_prediction(pred_df)

    return render_template(
        "home.html",
        prediction=explainability_result["prediction"],
        confidence_pct=explainability_result["confidence_pct"],
        top_drivers=explainability_result["top_drivers"],
        feature_contributions=explainability_result["feature_contributions"]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)