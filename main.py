from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Luna Women's Health Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    message_type: str  # "normal", "emergency", "out_of_domain"
    
# Global variables for model and tokenizer
model = None
tokenizer = None

class WomensHealthChatbot:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        # Emergency keywords
        self.emergency_keywords = {
            'suicide': '🆘 This seems serious. Please contact the National Suicide Prevention Lifeline at 988 or go to your nearest emergency room immediately.',
            'kill myself': '🆘 Please reach out for help immediately. National Suicide Prevention Lifeline: 988 or emergency services: 911.',
            'hurt myself': '🆘 Please contact a crisis helpline: National Suicide Prevention Lifeline 988 or Crisis Text Line: Text HOME to 741741.',
            'severe bleeding': '🚨 Heavy bleeding can be a medical emergency. Please seek immediate medical attention or call 911.',
            'severe pain': '🚨 Severe pain requires immediate medical evaluation. Please contact your healthcare provider or emergency services.',
            'emergency': '🚨 This sounds like a medical emergency. Please call emergency services immediately or go to your nearest emergency room.',
            'unconscious': '🚨 Loss of consciousness is a medical emergency. Call 911 immediately.',
            'overdose': '🚨 This is a medical emergency. Call Poison Control at 1-800-222-1222 or 911 immediately.'
        }
        
        # Women's health keywords
        self.health_keywords = [
            'pregnancy', 'period', 'menstrual', 'contraception', 'fertility',
            'breast', 'vaginal', 'uterus', 'ovary', 'hormone', 'pcos',
            'endometriosis', 'menopause', 'pap smear', 'gynecologist',
            'birth control', 'ovulation', 'cramps', 'discharge', 'infection',
            'health', 'pain', 'symptoms', 'doctor', 'medical', 'pregnant',
            'cycle', 'bleeding', 'contraceptive', 'reproductive', 'sex', 'sexual health','bleed',
            'menstruation', 'wellness', 'obstetrics', 'gynecology', 'vulva',
        ]
    
    def detect_emergency(self, message: str) -> Optional[str]:
        """Detect emergency situations"""
        message_lower = message.lower()
        for keyword, response in self.emergency_keywords.items():
            if keyword in message_lower:
                return response
        return None
    
    def is_women_health_related(self, message: str) -> bool:
        """Check if message is related to women's health"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.health_keywords)
    
    def clean_response(self, text: str) -> str:
        """Clean and format the complete response"""
        
        # Remove any unwanted prefixes or suffixes
        text = text.strip()
        
        # Remove common model artifacts
        unwanted_patterns = [
            "USER:", "DOCTOR:", "Human:", "Assistant:", "AI:",
            "Note:", "Disclaimer:", "Please note that"
        ]
        
        for pattern in unwanted_patterns:
            if text.startswith(pattern):
                text = text[len(pattern):].strip()
        
        # Clean up extra whitespace and newlines
        import re
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r' +', ' ', text)  # Remove multiple spaces
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Add medical disclaimer if response is substantial
        if len(text.split()) > 10:
            text += "\n\n💡 Please consult your healthcare provider for personalized advice."
        
        return text
    
    def generate_response(self, question: str) -> dict:
        """Generate response with safety checks"""
        
        # Emergency check
        emergency_response = self.detect_emergency(question)
        if emergency_response:
            return {
                "success": True,
                "response": emergency_response,
                "message_type": "emergency"
            }
        
        # Domain relevance check
        if not self.is_women_health_related(question):
            return {
                "success": True,
                "response": "🌸Hey there, I'm Luna a women's health specialist chatbot 💕 💖. I can help with questions about reproductive health, pregnancy, menstrual health, contraception, fertility, and other women's wellness topics🌈. Could you ask a women's health related question?✨",
                "message_type": "out_of_domain"
            }
        
        # Generate medical response
        try:
            prompt = f"USER: {question}\nDOCTOR:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,         # Total sequence length (prompt + response)
                    min_length=inputs['input_ids'].shape[1] + 20,  # Ensure minimum response length
                    temperature=0.7,        # Slightly higher for more natural responses
                    do_sample=True,
                    top_p=0.9,             # Allow more diverse vocabulary
                    top_k=40,              # Limit to top 40 tokens for quality
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    num_beams=1           # greedy search for consistency
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = response.split("DOCTOR:")[-1].strip()
            
            # Clean and format the complete response
            generated_response = self.clean_response(generated_response)
            
            return {
                "success": True,
                "response": generated_response,
                "message_type": "normal"
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"I apologize, but I'm having trouble generating a response right now. Please try again. Error: {str(e)}",
                "message_type": "error"
            }

# Initialize chatbot
chatbot_instance = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, chatbot_instance
    
    try:
        print("Loading model from Hugging Face...")
        
       
        model_name = "JCholder/womens-health-chatbot"
        
       
        
        print(f"Downloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize chatbot
        chatbot_instance = WomensHealthChatbot(model, tokenizer)
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a smaller model...")
        
        try:
            # Fallback to a smaller model
            model_name = "microsoft/DialoGPT-small"
            print(f"Loading fallback model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            chatbot_instance = WomensHealthChatbot(model, tokenizer)
            print("Fallback model loaded successfully!")
            
        except Exception as fallback_error:
            print(f"Fallback model also failed: {fallback_error}")
            raise

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    
    if chatbot_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = chatbot_instance.generate_response(request.question)
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": chatbot_instance is not None,
        "message": "Luna Women's Health Chatbot API is running"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Luna Women's Health Chatbot API 🌸",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)