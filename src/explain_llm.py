import os
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Load a lightweight LLM like Mistral (or use any compatible HF model)
#explainer = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", token=hf_token)
explainer = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    token=hf_token
)

def generate_explanation(image_label, confidence):
    prompt = (
        f"The AI model predicted that the uploaded image is a {image_label} one, with a confidence of {confidence:.2f}. "
        f"Explain in simple terms why AI might make such a prediction."
    )
    result = explainer(prompt, max_new_tokens=100, do_sample=True)[0]['generated_text']
    return result


if __name__ == "__main__":
    label = "Fake"
    confidence = 0.89
    explanation = generate_explanation(label, confidence)
    print("Explanation:")
    print(explanation)
