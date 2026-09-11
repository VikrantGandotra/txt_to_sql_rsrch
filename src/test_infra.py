import os
from config import settings

# 1. We must set these environment variables BEFORE importing Langfuse wrappers
# so the SDK knows exactly which project to write to.
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url

# 2. Use Langfuse's drop-in replacement for the OpenAI client
from langfuse.openai import OpenAI
from langfuse import Langfuse

def test_inference_and_tracing():
    print(f"Initializing OpenRouter client for model: {settings.model_name}...")
    
    # Initialize the Langfuse client explicitly so we can force a flush at the end
    langfuse_client = Langfuse()
    
    # 3. Configure the client to point to OpenRouter
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/mars-ablation", # Optional identifying header
            "X-Title": "MARS-SQL Serverless Ablation" 
        }
    )

    prompt = "Write a basic SQLite query to find the names of all employees who make more than $50,000."
    
    print("\nSending test query to Qwen2.5-Coder-7B-Instruct via OpenRouter...")
    try:
        # The 'name' parameter tags this specific trace in your Langfuse dashboard
        response = client.chat.completions.create(
            name="infra_setup_test", 
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "You are a highly capable SQL coding assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        print("\n=== Model Response ===")
        print(reply)
        print("======================\n")
        
        # 4. Critical: Ensure the trace is sent before the script exits
        print("Flushing telemetry data to Langfuse...")
        langfuse_client.flush()
        print("✅ Milestone 1 Complete: API successfully responded and trace recorded!")
        
    except Exception as e:
        print(f"❌ Error during API call: {e}")

if __name__ == "__main__":
    test_inference_and_tracing()