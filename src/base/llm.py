import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key từ file .env
load_dotenv()

def get_gemini_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.3,
    max_output_tokens: int = 1024,
    **kwargs
):
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in your environment or .env file")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs
    )

# Test (bỏ comment nếu muốn test trực tiếp)
# if __name__ == "__main__":
#     llm = get_gemini_llm()
#     print(llm.invoke("Xin chào, bạn là ai?"))