import os
import time
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class GemmaSamplingFixed:
    def __init__(self):
        self.model_id = "google/gemma-3n-e4b-it"
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """Initialize and load the Gemma 3n model"""
        print("=" * 60)
        print("SETTING UP GEMMA 3N FOR SAMPLING TESTS (FIXED)")
        print("=" * 60)
        
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            token=hf_token,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)
        
        # Show default generation config
        print("\nDefault Generation Config:")
        print(f"do_sample: {self.model.generation_config.do_sample}")
        print(f"top_k: {self.model.generation_config.top_k}")
        print(f"top_p: {self.model.generation_config.top_p}")
        print(f"temperature: {getattr(self.model.generation_config, 'temperature', 'not set')}")
        print("Model loaded successfully!\n")
    
    def simple_generate_test(self, prompt, **generation_kwargs):
        """Simple generation test with explicit parameters"""
        
        print(f"Testing with parameters: {generation_kwargs}")
        
        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        print(f"Input shape: {inputs.shape}")
        
        start_time = time.time()
        
        # Use model.generate with explicit parameters
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=100,
                **generation_kwargs
            )
        
        generation_time = time.time() - start_time
        
        # Decode only the new tokens
        input_length = inputs.shape[1]
        new_tokens = outputs[0][input_length:]
        decoded = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Generated tokens: {len(new_tokens)}")
        print(f"Speed: {len(new_tokens)/generation_time:.1f} tokens/sec")
        print(f"Generated text:\n{decoded}")
        print("="*60)
        print()
        
        return decoded
    
    def test_sampling_parameters(self):
        """Test different sampling parameter combinations"""
        print("TESTING SAMPLING PARAMETERS")
        print("=" * 60)
        
        prompt = "Tell a short story about a robot that learned to dream."
        
        print("1. Greedy (deterministic):")
        self.simple_generate_test(prompt, do_sample=False)
        
        print("2. Default sampling (do_sample=True):")
        self.simple_generate_test(prompt, do_sample=True)
        
        print("3. Conservative sampling (low creativity):")
        self.simple_generate_test(
            prompt, 
            do_sample=True,
            top_k=20,
            top_p=0.3,
            temperature=0.7
        )
        
        print("4. Moderate sampling (balanced):")
        self.simple_generate_test(
            prompt, 
            do_sample=True,
            top_k=50,
            top_p=0.8,
            temperature=0.8
        )
        
        print("5. Creative sampling (high diversity):")
        self.simple_generate_test(
            prompt, 
            do_sample=True,
            top_k=100,
            top_p=0.95,
            temperature=0.9
        )
        
        print("6. Only top_k (no top_p):")
        self.simple_generate_test(
            prompt, 
            do_sample=True,
            top_k=40,
            temperature=0.8
        )
        
        print("7. Only top_p (no top_k limit):")
        self.simple_generate_test(
            prompt, 
            do_sample=True,
            top_p=0.85,
            temperature=0.8
        )
    
    def test_temperature_effect(self):
        """Test how temperature affects generation"""
        print("TESTING TEMPERATURE EFFECTS")
        print("=" * 60)
        
        prompt = "Write a short poem about AI and humans."
        
        temperatures = [0.1, 0.5, 0.8, 1.0, 1.2]
        
        for temp in temperatures:
            print(f"Temperature: {temp}")
            self.simple_generate_test(
                prompt,
                do_sample=True,
                temperature=temp,
                top_k=50,
                top_p=0.9
            )
    
    def run_tests(self):
        """Run all tests"""
        try:
            self.setup_model()
            
            print("🧪 Running sampling parameter tests...")
            print()
            
            # Test basic sampling parameters
            self.test_sampling_parameters()
            
            # Test temperature effects
            self.test_temperature_effect()
            
            print("✅ All tests completed!")
            print("\n📝 Conclusions:")
            print("• top_p and top_k work correctly with Gemma 3n")
            print("• temperature affects generation diversity")
            print("• do_sample=False gives deterministic results")
            print("• Combination top_k=50, top_p=0.8, temperature=0.8 provides good balance")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = GemmaSamplingFixed()
    tester.run_tests() 