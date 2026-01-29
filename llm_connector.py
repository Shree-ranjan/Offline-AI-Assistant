import ollama
from typing import Dict, Any, List, Optional
import time


class LLMConnector:
    """
    Connector class for interfacing with local LLMs via Ollama.
    """
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.7, max_tokens: int = 2048):
        """
        Initialize the LLM connector.
        
        Args:
            model_name: Name of the model to use (must be pulled in Ollama)
            temperature: Temperature for generation (higher = more random)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_history = []
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response from the LLM based on the provided prompt.
        
        Args:
            prompt: Input prompt for the model
            system_prompt: Optional system prompt to guide behavior
            
        Returns:
            Generated response from the model
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add the user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Make the API call to Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating response: {error_msg}")
            
            # Check if it's a memory-related error
            if "memory" in error_msg.lower() or "out of memory" in error_msg.lower() or "requires more system memory" in error_msg.lower():
                return "Error: Insufficient memory to run this model. Try closing other applications or using a smaller model like phi3:mini or gemma:2b."
            
            return f"Error: Could not generate response. {error_msg}"
    
    def generate_stream(self, prompt: str, system_prompt: str = None):
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: Input prompt for the model
            system_prompt: Optional system prompt to guide behavior
            
        Yields:
            Streaming tokens/words from the model
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add the user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Make the streaming API call to Ollama
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating streaming response: {error_msg}")
            
            # Check if it's a memory-related error
            if "memory" in error_msg.lower() or "out of memory" in error_msg.lower() or "requires more system memory" in error_msg.lower():
                yield "Error: Insufficient memory to run this model. Try closing other applications or using a smaller model like phi3:mini or gemma:2b."
            else:
                yield f"Error: Could not generate response. {error_msg}"
    
    def chat(self, user_message: str, system_prompt: str = None) -> str:
        """
        Conduct a chat conversation, maintaining history.
        
        Args:
            user_message: Message from the user
            system_prompt: Optional system prompt to guide behavior
            
        Returns:
            Response from the model
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add chat history if available
            messages.extend(self.chat_history)
            
            # Add the current user message
            messages.append({"role": "user", "content": user_message})
            
            # Make the API call to Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": response['message']['content']})
            
            # Limit chat history to prevent context overflow
            if len(self.chat_history) > 20:  # Keep last 10 exchanges
                self.chat_history = self.chat_history[-20:]
            
            return response['message']['content']
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in chat: {error_msg}")
            
            # Check if it's a memory-related error
            if "memory" in error_msg.lower() or "out of memory" in error_msg.lower() or "requires more system memory" in error_msg.lower():
                return "Error: Insufficient memory to run this model. Try closing other applications or using a smaller model like phi3:mini or gemma:2b."
            
            return f"Error: Could not generate response. {error_msg}"
    
    def reset_chat_history(self):
        """Reset the chat history."""
        self.chat_history = []
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.
        
        Returns:
            List of available models
        """
        try:
            response = ollama.list()
            return response['models']
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a specific model exists in Ollama.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        models = self.list_available_models()
        try:
            # Handle different possible structures of model objects
            for model in models:
                if isinstance(model, dict):
                    # Try different possible keys for the model name
                    if 'name' in model and model['name'].startswith(model_name):
                        return True
                    elif 'model' in model and model['model'].startswith(model_name):
                        return True
            return False
        except Exception:
            # If there's any error accessing the model names, return False
            return False
    
    def pull_model(self, model_name: str):
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
        """
        try:
            print(f"Pulling model: {model_name}...")
            response = ollama.pull(model_name)
            print(f"Model {model_name} pulled successfully!")
            return response
        except Exception as e:
            print(f"Error pulling model {model_name}: {str(e)}")
            raise e


class AdvancedLLMConnector(LLMConnector):
    """
    Extended LLM connector with additional features like model switching and enhanced error handling.
    """
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.7, max_tokens: int = 2048):
        super().__init__(model_name, temperature, max_tokens)
        self.available_models = self.list_available_models()
        self.default_system_prompt = (
            "You are an AI assistant that helps users with their questions based on provided context. "
            "Be concise, accurate, and helpful. If the provided context doesn't contain the information "
            "needed to answer the question, say so."
        )
    
    def generate_with_fallback(self, prompt: str, system_prompt: str = None, fallback_models: List[str] = None) -> str:
        """
        Generate response with fallback to alternative models if primary fails.
        
        Args:
            prompt: Input prompt for the model
            system_prompt: Optional system prompt to guide behavior
            fallback_models: List of fallback model names to try
            
        Returns:
            Generated response from the model
        """
        if fallback_models is None:
            fallback_models = ["mistral", "phi3", "gemma"]
        
        models_to_try = [self.model_name] + fallback_models
        
        for model in models_to_try:
            try:
                print(f"Trying model: {model}")
                messages = []
                
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                else:
                    messages.append({"role": "system", "content": self.default_system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                )
                
                # If successful, update current model and return response
                self.model_name = model
                return response['message']['content']
                
            except Exception as e:
                print(f"Model {model} failed: {str(e)}")
                continue  # Try next model
        
        # If all models fail, return error message
        return "Error: All available models failed to generate a response."
    
    def set_model(self, model_name: str):
        """
        Change the model used by the connector.
        
        Args:
            model_name: Name of the new model to use
        """
        try:
            if self.check_model_exists(model_name):
                self.model_name = model_name
                print(f"Model changed to: {model_name}")
            else:
                raise ValueError(f"Model {model_name} not found. Please pull it first.")
        except KeyError as e:
            # Handle cases where the model structure is different than expected
            print(f"Error checking model existence: {str(e)}")
            # Assume the model exists to avoid blocking the user
            self.model_name = model_name
            print(f"Model changed to: {model_name} (verification skipped due to API differences)")
    
    def generate_structured(self, prompt: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Generate structured output in a specific format.
        
        Args:
            prompt: Input prompt for the model
            output_format: Desired output format ('text', 'json', 'list', 'dict')
            
        Returns:
            Structured response from the model
        """
        system_prompts = {
            "json": "You are a helpful AI assistant. Return only valid JSON format as specified by the user. Do not include any explanations or markdown formatting.",
            "list": "You are a helpful AI assistant. Return your response as a bulleted list. Start each item with '- '. Do not include any introductory text.",
            "dict": "You are a helpful AI assistant. Return your response as a dictionary with relevant keys and values. Do not include any explanatory text."
        }
        
        system_prompt = system_prompts.get(output_format, self.default_system_prompt)
        
        response = self.generate(prompt, system_prompt)
        
        result = {
            'raw_response': response,
            'format': output_format,
            'processed': self._process_output(response, output_format)
        }
        
        return result
    
    def _process_output(self, raw_response: str, output_format: str) -> Any:
        """
        Process the raw response into the desired format.
        
        Args:
            raw_response: Raw response from the model
            output_format: Desired output format
            
        Returns:
            Processed response in the specified format
        """
        if output_format == "json":
            import json
            try:
                # Try to extract JSON from the response if it contains other text
                start_idx = raw_response.find('{')
                end_idx = raw_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = raw_response[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # If no JSON delimiters found, return the raw response
                    return raw_response
            except json.JSONDecodeError:
                return raw_response
        elif output_format == "list":
            lines = raw_response.split('\n')
            items = [line.strip()[2:] for line in lines if line.strip().startswith('- ')]
            return items
        elif output_format == "dict":
            import json
            try:
                # Attempt to parse as JSON
                return json.loads(raw_response)
            except json.JSONDecodeError:
                # If not valid JSON, try to create a dict from key-value pairs
                lines = raw_response.split('\n')
                result_dict = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result_dict[key.strip()] = value.strip()
                return result_dict
        else:
            return raw_response