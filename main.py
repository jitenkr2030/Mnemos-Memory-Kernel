"""
Main entry point for Mnemos Memory Kernel.

This module provides the primary entry point for running the Mnemos
memory kernel as a standalone service or embedded library.
"""

import uvicorn
from .config import MnemosConfig
from .kernel import MnemosKernel
from .api import app


def create_kernel(config: Optional[MnemosConfig] = None) -> MnemosKernel:
    """
    Create and configure a Mnemos kernel instance.
    
    Args:
        config: Optional configuration. If not provided, uses defaults
               and environment variables.
    
    Returns:
        Configured MnemosKernel instance
    """
    if config is None:
        config = MnemosConfig.from_env()
    
    kernel = MnemosKernel(
        storage_dir=config.storage_dir,
        enable_llm_classification=config.llm_enabled,
        llm_provider=config.llm_provider
    )
    
    return kernel


def run_server(config: Optional[MnemosConfig] = None) -> None:
    """
    Run the Mnemos API server.
    
    This starts the FastAPI server with the provided configuration.
    
    Args:
        config: Optional configuration. If not provided, uses defaults.
    """
    if config is None:
        config = MnemosConfig.from_env()
    
    uvicorn.run(
        "src.api.api:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.log_level == "debug"
    )


def main() -> None:
    """Main entry point when run as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mnemos Memory Kernel - Personal Knowledge Management"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "shell"],
        default="server",
        help="Run mode: server (API) or shell (interactive)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for the API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server"
    )
    parser.add_argument(
        "--storage-dir",
        default="./data",
        help="Directory for storing memory data"
    )
    
    args = parser.parse_args()
    
    if args.mode == "server":
        config = MnemosConfig(
            storage_dir=args.storage_dir,
            api_host=args.host,
            api_port=args.port
        )
        run_server(config)
    elif args.mode == "shell":
        # Simple interactive shell for testing
        print("Mnemos Interactive Shell")
        print("Type 'exit' to quit")
        
        kernel = create_kernel(MnemosConfig(storage_dir=args.storage_dir))
        
        while True:
            try:
                text = input("\nEnter transcript text: ")
                if text.lower() in ("exit", "quit", "q"):
                    break
                
                from .kernel import TranscriptInput
                transcript = TranscriptInput(text=text)
                memory = kernel.ingest(transcript)
                print(f"Created memory: {memory.intent.value} (confidence: {memory.confidence:.2f})")
                print(f"Memory ID: {memory.id}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
