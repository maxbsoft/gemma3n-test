import os
import time
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class Gemma3nTester:
    def __init__(self):
        self.model_id = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """Initialize and load the Gemma 3n model"""
        print("=" * 60)
        print("SETTING UP GEMMA 3N MODEL")
        print("=" * 60)
        
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        print("Loading model...")
        start_time = time.time()
        
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            # low_cpu_mem_usage=True,
            # trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory allocated: {memory_allocated:.2f} GB")
        
        print("Model setup complete!\n")
    
    def generate_with_speed_test(self, messages, max_new_tokens=200, use_sampling=True):
        """Generate text with proper sampling parameters and speed measurement"""
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        start_time = time.time()
        
        with torch.inference_mode():
            if use_sampling:
                # Use optimized sampling parameters
                # generation = self.model.generate(
                #     input_ids=inputs["input_ids"],
                #     attention_mask=inputs.get("attention_mask", None),
                #     max_new_tokens=max_new_tokens,
                #     do_sample=True,
                #     top_k=50,
                #     top_p=0.8,
                #     temperature=0.8,
                #     pad_token_id=self.processor.tokenizer.eos_token_id,
                #     use_cache=True
                # )

                generation = self.model.generate(
                    **inputs,
                    # attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=64,
                    top_p=0.95,
                    temperature=1.0,
                    min_p=0.0,
                    # pad_token_id=self.processor.tokenizer.eos_token_id,
                    # use_cache=True
                )
            else:
                # Greedy decoding - no sampling parameters
                # generation = self.model.generate(
                #     input_ids=inputs["input_ids"],
                #     attention_mask=inputs.get("attention_mask", None),
                #     max_new_tokens=max_new_tokens,
                #     do_sample=False,
                #     pad_token_id=self.processor.tokenizer.eos_token_id,
                #     use_cache=True
                # )

                generation = self.model.generate(
                    **inputs,
                    # attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=64,
                    top_p=0.95,
                    temperature=1.0,
                    min_p=0.0,
                    # pad_token_id=self.processor.tokenizer.eos_token_id,
                    # use_cache=True
                )
            
            generation = generation[0][input_len:]
        
        generation_time = time.time() - start_time
        output_tokens = len(generation)
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        return decoded, tokens_per_second, generation_time, output_tokens
    
    def test_text_english(self):
        """Test 1: English text generation"""
        print("=" * 60)
        print("TEST 1: ENGLISH TEXT GENERATION")
        print("=" * 60)
        
        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": "Write a creative short story about a robot discovering emotions. Make it engaging and detailed."}]
            }
        ]
        
        print("Prompt: Write a creative short story about a robot discovering emotions.")
        
        decoded, tokens_per_second, generation_time, output_tokens = self.generate_with_speed_test(
            messages, max_new_tokens=200, use_sampling=True
        )
        
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Output tokens: {output_tokens}")
        print(f"Speed: {tokens_per_second:.1f} tokens/second")
        print(f"\nGenerated text:\n{decoded}")
        print("\n" + "="*60 + "\n")
        
        return tokens_per_second
    
    def test_text_ukrainian(self):
        """Test 2: Ukrainian text generation"""
        print("=" * 60)
        print("TEST 2: UKRAINIAN TEXT GENERATION")
        print("=" * 60)
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Розкажи коротку історію про космічного мандрівника, який знайшов нову планету. Пиши українською мовою."}]
            }
        ]
        
        print("Prompt: Tell a short story about a space traveler who found a new planet (in Ukrainian)")
        
        decoded, tokens_per_second, generation_time, output_tokens = self.generate_with_speed_test(
            messages, max_new_tokens=200, use_sampling=True
        )
        
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Output tokens: {output_tokens}")
        print(f"Speed: {tokens_per_second:.1f} tokens/second")
        print(f"\nGenerated text:\n{decoded}")
        print("\n" + "="*60 + "\n")
        
        return tokens_per_second
    
    def create_test_image(self):
        """Create a simple test image"""
        if not os.path.exists("test_image.png"):
            print("Creating test image...")
            
            # Create a simple colored image with pattern
            size = (512, 512)
            img = Image.new('RGB', size, (100, 150, 200))
            
            # Add some pattern
            pixels = img.load()
            for x in range(0, size[0], 20):
                for y in range(0, size[1], 20):
                    if (x // 20 + y // 20) % 2 == 0:
                        for i in range(10):
                            for j in range(10):
                                if x + i < size[0] and y + j < size[1]:
                                    pixels[x + i, y + j] = (255, 255, 255)
            
            img.save("test_image.png")
            print("Test image created!")
    
    def test_image_processing(self):
        """Test 3: Image description"""
        print("=" * 60)
        print("TEST 3: IMAGE PROCESSING")
        print("=" * 60)
        
        self.create_test_image()
        
        try:
            img = Image.open("test_image.png")
            print(f"Processing image size: {img.size}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image in detail. What patterns and colors do you see?"}
                    ]
                }
            ]
            
            decoded, tokens_per_second, generation_time, output_tokens = self.generate_with_speed_test(
                messages, max_new_tokens=150, use_sampling=True
            )
            
            print(f"Processing time: {generation_time:.2f} seconds")
            print(f"Output tokens: {output_tokens}")
            print(f"Speed: {tokens_per_second:.1f} tokens/second")
            print(f"Description: {decoded}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
        
        print("\n" + "="*60 + "\n")
    
    def test_performance_comparison(self):
        """Test 4: Compare greedy vs sampling performance"""
        print("=" * 60)
        print("TEST 4: PERFORMANCE COMPARISON")
        print("=" * 60)
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Explain quantum computing in simple terms."}]
            }
        ]
        
        print("Testing Greedy Decoding (deterministic):")
        decoded_greedy, speed_greedy, time_greedy, tokens_greedy = self.generate_with_speed_test(
            messages, max_new_tokens=100, use_sampling=False
        )
        print(f"Greedy - Speed: {speed_greedy:.1f} tokens/sec, Time: {time_greedy:.2f}s")
        
        print("\nTesting Optimized Sampling:")
        decoded_sampling, speed_sampling, time_sampling, tokens_sampling = self.generate_with_speed_test(
            messages, max_new_tokens=100, use_sampling=True
        )
        print(f"Sampling - Speed: {speed_sampling:.1f} tokens/sec, Time: {time_sampling:.2f}s")
        
        print(f"\nSpeedup with sampling: {speed_sampling/speed_greedy:.2f}x")
        print("\n" + "="*60 + "\n")
        
        return speed_greedy, speed_sampling
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("STARTING GEMMA 3N PERFORMANCE TESTS")
        print("=" * 60)
        
        try:
            self.setup_model()
            
            # Run tests
            english_speed = self.test_text_english()
            ukrainian_speed = self.test_text_ukrainian()
            self.test_image_processing()
            speed_greedy, speed_sampling = self.test_performance_comparison()
            
            # Summary
            print("=" * 60)
            print("PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"English text generation: {english_speed:.1f} tokens/second")
            print(f"Ukrainian text generation: {ukrainian_speed:.1f} tokens/second")
            print(f"Greedy decoding: {speed_greedy:.1f} tokens/second")
            print(f"Optimized sampling: {speed_sampling:.1f} tokens/second")
            print(f"Sampling speedup: {speed_sampling/speed_greedy:.2f}x")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run tests"""
    tester = Gemma3nTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
