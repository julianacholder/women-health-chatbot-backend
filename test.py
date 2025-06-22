import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_chat(question):
    """Test the chat endpoint"""
    url = f"{BASE_URL}/chat"
    payload = {"question": question}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"\n🤖 Question: {question}")
            print(f"📝 Response: {result['response']}")
            print(f"🏷️  Type: {result['message_type']}")
            print("-" * 80)
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"🏥 Health Check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def interactive_chat():
    """Interactive chat mode"""
    print("\n💬 Interactive Luna Chat Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        question = input("\n👤 You: ").strip()
        
        if question.lower() == 'quit':
            print("👋 Goodbye!")
            break
            
        if not question:
            print("❓ Please ask a question")
            continue
            
        test_chat(question)

if __name__ == "__main__":
    print("🚀 Testing Luna Women's Health Chatbot API")
    print("=" * 60)
    
    # Test health endpoint
    test_health()
    
    # Test various questions
    test_questions = [
        # Normal women's health questions
        "What are the symptoms of PCOS?",
        "How can I manage menstrual cramps?",
        "What should I know about pregnancy nutrition?",
        "Is irregular periods normal?",
        "What are the signs of early pregnancy?",
        
        # Emergency situations
        "I'm having severe bleeding and feel faint",
        "I'm having thoughts of hurting myself",
        
        # Out of domain
        "How to cook pasta?",
        "What's the weather like?",
        "How to fix my car?"
    ]
    
    print(f"\n🧪 Running {len(test_questions)} test questions...")
    
    for question in test_questions:
        test_chat(question)
    
    # Interactive mode
    print(f"\n🎯 Test Results Complete!")
    choice = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
    
    if choice == 'y':
        interactive_chat()