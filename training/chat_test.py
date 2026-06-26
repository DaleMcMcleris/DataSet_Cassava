from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuggingFaceTB/SmolLM2-135M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("🤖 Telecom AI Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"""You are a helpful telecom assistant.

User: {user_input}
Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # clean output (show only assistant part)
    answer = response.split("Assistant:")[-1].strip()

    print("\nBot:", answer, "\n")