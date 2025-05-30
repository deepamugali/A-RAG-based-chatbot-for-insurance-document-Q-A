from flask import Flask, request, jsonify, render_template
from rag_bot import get_answer

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please provide a valid question"}), 400
    answer = get_answer(question)
    return jsonify({"answer": answer})

@app.route("/suggestions")
def suggestions():
    import json
    with open("suggested_questions.json", "r") as f:
        questions = json.load(f)
    return jsonify(questions)


if __name__ == "__main__":
    app.run(debug=True)
