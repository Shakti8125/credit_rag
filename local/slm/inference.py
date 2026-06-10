import logging
from pathlib import Path
from typing import List, Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LocalModelInference:
    """
    Manages low-level lifecycle loading and predictive execution of quantized 
    GGUF model binaries on local workstation edge hardware configurations.
    """
    def __init__(self, model_path: str, ctx_size: int = 2048, gpu_layers: int = 0) -> None:
        """
        Initializes the model architecture framework.
        
        Args:
            model_path: Local disk path directly referencing the GGUF binary.
            ctx_size: Maximum structural context token width window.
            gpu_layers: Total number of network layers to offload to hardware acceleration.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            error_msg = f"Quantized GGUF model binary not located at specified layout: {self.model_path}"
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Binding llama-cpp engine structure to target model: {self.model_path.name}")
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=ctx_size,
                n_gpu_layers=gpu_layers,
                n_threads=4,       # Optimized assignment for generic CPU core compute architectures
                verbose=False       # Suppresses noisy native C++ standard console printouts
            )
            logger.info("Local SLM engine bound and verified in memory.")
        except Exception as e:
            logger.critical(f"Critical execution error initializing GGUF engine: {e}", exc_info=True)
            raise RuntimeError(f"GGUF Initialization Failure: {e}")

    def run_inference(self, prompt: str, max_tokens: int = 16, stop_tokens: Optional[List[str]] = None) -> str:
        """
        Executes a deterministic non-streaming forward prediction pass over local hardware.
        """
        stops = stop_tokens or ["<|end|>", "\n"]
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,    # Zero temperature turns off random sampling for reliable routing
                stop=stops
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Inference execution sequence encountered a runtime error: {e}", exc_info=True)
            raise RuntimeError(f"SLM Generation Error: {e}")