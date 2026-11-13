
import ollama

# Send a prompt to the local AI model
response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "What is Artificial Intelligence? Explain in simple words."
        }
    ]
)

# Display the AI response
print("\nLLM Response:")
print(response.message.content)
