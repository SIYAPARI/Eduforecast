# EduForecast

EduForecast is an explainable machine learning project designed to predict a student’s academic performance based on key educational and demographic factors.

This project combines:
- a trained ML model
- a Flask web application
- explainable prediction insights
- an easy-to-use interface for entering student details

The goal is not just to predict outcomes, but to help users understand why a prediction was made.

## Why this project?

Academic performance prediction is useful for:
- students and parents
- teachers and academic advisors
- school administrators
- educational decision support systems

However, a prediction becomes much more valuable when it explains the reasons behind it. That is where EduForecast focuses on explainability.

## Project Overview

EduForecast takes input such as:
- gender
- race/ethnicity
- parental level of education
- lunch type
- test preparation course
- reading score
- writing score

It then predicts a student’s performance using a machine learning pipeline and provides an interpretable explanation of the result.

## Core Idea

This project demonstrates an end-to-end ML workflow:
1. collect structured student data
2. clean and preprocess the data
3. train a machine learning model
4. use a Flask app to accept user input
5. generate a prediction
6. explain which features influenced the output

## Explainable AI Focus

This project is built with an explainability-first mindset.

Planned and future capabilities include:
- feature importance breakdown
- highlight the strongest contributing factors
- explain which inputs drove the prediction
- confidence/uncertainty indicator
- “what-if” simulation for score changes
- interpretable output for non-technical users

This makes the system more transparent, trustworthy, and useful in real educational settings.

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- Jinja2 Templates
- HTML/CSS

## Project Structure

```text
ml_full_project/
├── application.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── pipeline/
│       └── predict_pipeline.py
├── templates/
│   ├── home.html
│   └── index.html
└── notebooks/
    └── (optional for experimentation)
```

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd ml_full_project
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
```

On Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:
```bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

If the requirements file is not present yet, install the main dependencies manually:
```bash
pip install flask pandas numpy scikit-learn
```

## Run the Application

From the project root:
```bash
python application.py
```

Then open:
```text
http://127.0.0.1:5000
```

## How It Works

1. The user enters student information through a form
2. The form data is captured in `application.py`
3. The input is converted into a DataFrame
4. A prediction pipeline preprocesses the data and runs the model
5. The predicted result is returned to the web interface

## Example Workflow

A teacher or academic advisor can enter:
- Student gender
- Parent education level
- Reading score
- Writing score
- Test preparation status

The app then returns a prediction for the student’s academic performance.

## Explainability Goals

The app aims to answer questions like:
- What factors influenced the prediction most?
- Which inputs matter the most?
- How much does writing score affect the result?
- How would the prediction change if a student improved in a certain area?

These questions are central to making machine learning practical and trustworthy in education.

## Planned Enhancements

- feature importance dashboard
- SHAP or permutation-based explanations
- interactive “what-if” scenario panel
- visual charts for model interpretation
- improved dashboard UI
- model version tracking
- batch prediction support

## Future Direction

EduForecast is intended to evolve from a simple prediction app into a more complete educational analytics tool that helps users understand both the result and the reasoning behind it.

## License

This project is intended for learning, experimentation, and academic demonstration.

## Acknowledgements

This project is inspired by common student performance datasets and educational prediction use cases.

## Project Name

EduForecast

A name that reflects:
- education
- prediction
- forecasting
- data-driven insights
- clarity and decision support

## Final Note

EduForecast is more than a prediction app — it is a step toward explainable AI in education, designed to make model outputs understandable and useful for real-world decision-making.