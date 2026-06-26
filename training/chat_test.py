import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Optional: Suppress the HF token warning if you are strictly loading local weights
# os.environ["HF_TOKEN"] = "your_token_here"

# 1. Define your model paths
base_model_name = "HuggingFaceTB/SmolLM2-135M"
adapter_path = "./telecom.json" 

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Important fix for some models (like SmolLM)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 2. Load the fine-tuned LoRA adapter weights
print(f"Loading PEFT adapter from '{adapter_path}'...")
try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
except Exception as e:
    print(f"\n❌ Error loading adapter from '{adapter_path}':")
    print(e)
    print("\nFallback: Running script with the RAW base model instead.\n")
    model = base_model

print("\n🤖 Telecom AI Chatbot ready! Type 'exit' to quit.\n")

# 3. Interactive Chat Loop
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_input.strip():
        continue

    # Structured prompt matching your fine-tuning format
    prompt = f"### Instruction:\n{user_input}\n\n### Response:"

    # Tokenize input and move to the correct device (CPU/GPU)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode entire output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean response extraction
    if "### Response:" in response:
        answer = response.split("### Response:")[-1].strip()
    else:
        # If model overwrote the structure, strip out the initial prompt
        answer = response.replace(prompt, "").strip()

    print(f"\nBot: {answer}\n")