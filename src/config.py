from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

# Resolves: .../mars_ablation/src/config.py -> parent is 'src' -> parent.parent is 'mars_ablation'
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    openrouter_api_key: str = ""
    vllm_base_url: str = ""
    vllm_api_key: SecretStr = SecretStr("")
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str 

    
    # model_name: str = "openrouter/qwen/qwen-2.5-7b-instruct"
    # model_name: str = "google/gemini-2.5-flash"
    # model_name = "openrouter/qwen/qwen-2.5-7b-instruct"
    model_name: str = "openrouter/qwen/qwen-2.5-7b-instruct"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH, 
        env_file_encoding='utf-8', 
        extra='ignore',
        hide_input_in_errors=True,
    )

settings = Settings()
