"""
LangChain integration examples for Gemma 3n API
"""
import asyncio
import sys
from pathlib import Path

# Add the server directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from integrations.langchain_integration import (
    GemmaLLM, GemmaChatModel, create_gemma_llm, create_gemma_chat_model,
    create_multimodal_message, call_multimodal_api
)

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory

def test_basic_llm():
    """Test basic LLM functionality"""
    print("=== Basic LLM Test ===")
    
    llm = create_gemma_llm(
        model="gemma-3n-e2b-quantized",
        temperature=0.7,
        max_tokens=100
    )
    
    prompt = "Write a short poem about machine learning"
    response = llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

async def test_async_llm():
    """Test asynchronous LLM"""
    print("\n=== Async LLM Test ===")
    
    llm = create_gemma_llm(
        model="gemma-3n-e2b-quantized",
        temperature=0.8,
        max_tokens=120
    )
    
    prompt = "Explain the concept of neural networks in simple terms"
    response = await llm.acall(prompt)  # type: ignore
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

def test_chat_model():
    """Test chat model functionality"""
    print("\n=== Chat Model Test ===")
    
    chat_model = create_gemma_chat_model(
        model="gemma-3n-e2b-quantized",
        temperature=0.7,
        max_tokens=150
    )
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant specializing in technology."),
        HumanMessage(content="What are the key differences between AI and machine learning?")
    ]
    
    response = chat_model(messages)
    print(f"System: {messages[0].content}")
    print(f"Human: {messages[1].content}")
    print(f"AI: {response.content}")

async def test_async_chat_model():
    """Test asynchronous chat model"""
    print("\n=== Async Chat Model Test ===")
    
    chat_model = create_gemma_chat_model(
        model="gemma-3n-e2b-quantized",
        temperature=0.8,
        max_tokens=100
    )
    
    messages = [
        HumanMessage(content="Write a haiku about coding")
    ]
    
    response = await chat_model.agenerate([messages])  # type: ignore
    print(f"Prompt: {messages[0].content}")
    print(f"Response: {response.generations[0].message.content}")  # type: ignore

def test_prompt_template():
    """Test with LangChain prompt templates"""
    print("\n=== Prompt Template Test ===")
    
    llm = create_gemma_llm(
        model="gemma-3n-e2b-quantized",
        temperature=0.7,
        max_tokens=120
    )
    
    template = """
    You are an expert in {subject}. 
    Please explain {topic} in simple terms that a beginner can understand.
    
    Topic: {topic}
    Explanation:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["subject", "topic"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.run(
        subject="artificial intelligence",
        topic="deep learning"
    )
    
    print(f"Subject: artificial intelligence")
    print(f"Topic: deep learning")
    print(f"Response: {response}")

def test_conversation_chain():
    """Test conversation chain with memory"""
    print("\n=== Conversation Chain Test ===")
    
    llm = create_gemma_llm(
        model="gemma-3n-e2b-quantized",
        temperature=0.8,
        max_tokens=100
    )
    
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # First exchange
    response1 = conversation.predict(input="Hi, I'm learning about AI. Can you help me?")
    print(f"Human: Hi, I'm learning about AI. Can you help me?")
    print(f"AI: {response1}")
    
    # Second exchange (should remember context)
    response2 = conversation.predict(input="What's the difference between supervised and unsupervised learning?")
    print(f"\nHuman: What's the difference between supervised and unsupervised learning?")
    print(f"AI: {response2}")

async def test_multimodal_direct():
    """Test direct multimodal API calls"""
    print("\n=== Direct Multimodal API Test ===")
    
    # Create a multimodal message
    message = create_multimodal_message(
        text="Describe this image in detail",
        image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    )
    
    try:
        response = await call_multimodal_api(
            messages=[message],
            model="gemma-3n-e2b-quantized",
            max_tokens=150
        )
        print(f"Image URL: {message['content'][1]['url']}")
        print(f"Description: {response}")
    except Exception as e:
        print(f"Error in multimodal call: {e}")

def test_different_presets():
    """Test different generation presets"""
    print("\n=== Generation Presets Test ===")
    
    presets = ["creative", "balanced", "precise", "deterministic"]
    prompt = "Write a short story about a robot's first day at work"
    
    for preset in presets:
        print(f"\n--- {preset.upper()} PRESET ---")
        
        llm = create_gemma_llm(
            model="gemma-3n-e2b-quantized",
            preset=preset,
            max_tokens=100
        )
        
        response = llm(prompt)
        print(f"Response: {response[:200]}...")  # Truncate for readability

def test_chain_of_thought():
    """Test chain of thought reasoning"""
    print("\n=== Chain of Thought Test ===")
    
    chat_model = create_gemma_chat_model(
        model="gemma-3n-e2b-quantized",
        temperature=0.3,  # Lower temperature for more focused reasoning
        max_tokens=200
    )
    
    messages = [
        SystemMessage(content="You are a logical reasoning assistant. Think step by step."),
        HumanMessage(content="""
        Problem: A farmer has chickens and rabbits. In total, there are 35 heads and 94 legs. 
        How many chickens and how many rabbits does the farmer have?
        
        Please solve this step by step.
        """)
    ]
    
    response = chat_model(messages)
    print(f"Problem: Chickens and rabbits puzzle")
    print(f"Response: {response.content}")

async def main():
    """Run all examples"""
    print("🧪 Testing LangChain Integration with Gemma 3n API")
    print("Make sure the API server is running on http://localhost:8000")
    
    # Test basic functionality
    test_basic_llm()
    await test_async_llm()
    test_chat_model()
    await test_async_chat_model()
    
    # Test advanced features
    test_prompt_template()
    test_conversation_chain()
    await test_multimodal_direct()
    test_different_presets()
    test_chain_of_thought()
    
    print("\n🎉 All LangChain tests completed!")

if __name__ == "__main__":
    # Check if server is running
    import requests
    try:
        response = requests.get("http://localhost:8000/v1/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding correctly")
            print("Please start the server with: python main.py")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ API server is not running")
        print("Please start the server with: python main.py")
        exit(1)
    
    print("✅ API server is running, starting LangChain tests...\n")
    
    asyncio.run(main()) 