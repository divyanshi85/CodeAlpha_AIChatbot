from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# FAQs
questions = [
    "What is your name?",
    "What is refund policy?",
    "How to contact support?",
    "What services do you provide?",
    "Where are you located?"
]

answers = [
    "I am your AI chatbot 🤖",
    "Refund is available within 7 days.",
    "You can contact us at support@gmail.com",
    "We provide AI and web development services.",
    "We are an online service."
]

# NLP Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    max_score = similarity.max()
    index = similarity.argmax()

    # 🔥 Add threshold
    if max_score < 0.3:
        return "Sorry, I didn't understand that. 🤔"

    return answers[index]
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    bot_reply = get_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)