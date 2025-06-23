from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import os
from typing import Optional
import uvicorn
import psutil

# Initialize FastAPI app
app = FastAPI(title="Luna Women's Health Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    message_type: str

# Global variables for model and tokenizer
model = None
tokenizer = None

def log_memory_usage(stage: str):
    """Log current memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage at {stage}: {memory_mb:.2f} MB")

def cleanup_memory():
    """Force garbage collection and clear cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class WomensHealthChatbot:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if self.model:
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
            'menstruation', 'wellness', 'obstetrics', 'gynecology', 'vulva', 'intercourse', 
            'fertility awareness', 'prenatal', 'postnatal', 'hysterectomy', 'fibroids', 
            'cervical health', 'vaginitis', 'premenstrual syndrome', 'pms', 'pelvic pain'
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
    
    def smart_truncate(self, text: str) -> str:
        """Truncate to complete sentences with medical disclaimer"""
        
        # Split into sentences
        sentences = text.split('. ')
        
        # Find optimal cutoff (aim for 2-3 complete sentences)
        word_count = 0
        selected_sentences = []
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= 100:  # Slightly increased limit
                selected_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        # Join sentences
        if selected_sentences:
            result = '. '.join(selected_sentences)
            if not result.endswith('.'):
                result += '.'
        else:
            # Fallback: take first sentence
            result = sentences[0] + '.'
        
        # Add disclaimer
        result += "\n\n💡 Please consult your healthcare provider for personalized advice."
        
        return result
    
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
                "response": "🌸 Hey there, I'm Luna, a women's health specialist chatbot 💕. I can help with questions about reproductive health, pregnancy, menstrual health, contraception, fertility, and other women's wellness topics🌈. Could you ask a women's health related question?✨",
                "message_type": "out_of_domain"
            }
        
        # Generate medical response
        try:
            log_memory_usage("before generation")
            
            prompt = f"USER: {question}\nDOCTOR:"
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=200,  # Limit input length
                padding=False
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120,     # Balanced length
                    min_new_tokens=15,      # Ensure minimum response
                    temperature=0.8,        # Good creativity
                    do_sample=True,
                    top_p=0.85,            # Focused sampling
                    top_k=35,              # Reduce choices for speed
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    use_cache=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = response.split("DOCTOR:")[-1].strip()
            
            # Smart truncation to complete sentences
            generated_response = self.smart_truncate(generated_response)
            
            # Clean up memory after generation
            cleanup_memory()
            log_memory_usage("after generation")
            
            return {
                "success": True,
                "response": generated_response,
                "message_type": "normal"
            }
            
        except Exception as e:
            print(f"Generation error: {e}")
            cleanup_memory()
            return {
                "success": False,
                "response": f"I apologize, but I'm having trouble generating a response right now. Please try again.",
                "message_type": "error"
            }

# Initialize chatbot
chatbot_instance = None

@app.on_event("startup")
async def load_model():
    """Load model on startup with optimizations"""
    global model, tokenizer, chatbot_instance
    
    try:
        log_memory_usage("startup")
        print("Loading optimized model from Hugging Face...")
        
        model_name = "JCholder/womens-health-chatbot"
        
        print(f"Downloading {model_name}...")
        
        # Load tokenizer first (smaller memory footprint)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,  # Use fast tokenizer for better performance
            padding_side="left"
        )
        
        log_memory_usage("after tokenizer")
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Half precision saves ~50% memory
            low_cpu_mem_usage=True,     # Reduce memory usage during loading
            device_map="cpu",           # Force CPU usage (Railway doesn't have GPU on free tier)
            trust_remote_code=True,     # Allow custom model code
            use_cache=True             # Enable KV cache for faster generation
        )
        
        log_memory_usage("after model loading")
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Force garbage collection
        cleanup_memory()
        log_memory_usage("after cleanup")
        
        # Initialize chatbot
        chatbot_instance = WomensHealthChatbot(model, tokenizer)
        
        print("✅ Model loaded successfully with optimizations!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Attempting to load fallback model...")
        
        try:
            # Fallback to smaller model
            model_name = "microsoft/DialoGPT-medium"
            print(f"Loading fallback model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            cleanup_memory()
            chatbot_instance = WomensHealthChatbot(model, tokenizer)
            print("✅ Fallback model loaded successfully!")
            
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {fallback_error}")
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
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with memory info"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        "status": "healthy",
        "model_loaded": chatbot_instance is not None,
        "memory_usage_mb": round(memory_mb, 2),
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)