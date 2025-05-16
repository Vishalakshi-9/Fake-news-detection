from flask import Flask, render_template, request
import pickle
from newspaper import Article

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to extract article text from URL
def extract_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title + " " + article.text
    except Exception as e:
        print(f"Error extracting article: {e}")
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    article_text = extract_from_url(url)

    if not article_text:
        return render_template("index.html", prediction="❌ Unable to extract content from the provided URL.")

    vec_input = vectorizer.transform([article_text])
    prediction = model.predict(vec_input)[0]

    return render_template("index.html", prediction=f"📰 Prediction: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
