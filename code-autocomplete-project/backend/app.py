from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load the fine-tuned GPT-2 model and tokenizer
model_path = "./model/fine-tuned-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

@app.route("/generate", methods=["POST"])
def generate_code():
    data = request.json
    input_text = data.get("input", "")

    # Tokenize input text
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate code
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Decode generated text
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"generated_code": generated_code})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)