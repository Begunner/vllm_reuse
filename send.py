import asyncio
from openai import AsyncOpenAI
import time

VLLM_API_BASE = "http://localhost:8000/v1" 

MODEL_NAME = "/mnt/model/Qwen3-14B/" 

PROMPTS = [
    "Once upon a time, ",
]

async def async_generate_completion(client: AsyncOpenAI, prompt: str, index: int):
    print(f"[{index}] Start: {prompt[:20]}...")
    start_time = time.time()
    
    try:
        response = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt, 
            max_tokens=11,
            temperature=0.7,
        )

        end_time = time.time()
        
        if response.choices:
            content = response.choices[0].text
            print(f"[{index}] Response (time: {end_time - start_time:.2f}s):")
            print(f"[{index}] Content: {content.strip()[:60]}...")
            
            usage = response.usage
            if usage:
                print(f"[{index}] Tokens Used: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")
        else:
            print(f"[{index}] No response from model.")

    except Exception as e:
        end_time = time.time()
        print(f"[{index}] Error (time: {end_time - start_time:.2f}s): {e}")

async def main():
    """
    Main function to create an asynchronous client and use asyncio.gather to send requests concurrently.
    """
    print(f"Connected to API Base: {VLLM_API_BASE}")
    print(f"Using model: {MODEL_NAME}")
    print("---")
    
    client = AsyncOpenAI(
        api_key="sk-not-required",
        base_url=VLLM_API_BASE,
        timeout=120.0
    )

    tasks = []
    for i, prompt in enumerate(PROMPTS):
        task = async_generate_completion(client, prompt, i + 1)
        tasks.append(task)
    
    start_total_time = time.time()
    
    print(f"Sending {len(tasks)} requests concurrently...")
    await asyncio.gather(*tasks)

    end_total_time = time.time()
    print("\nAll tasks completed")
    print(f"Total time: {end_total_time - start_total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())