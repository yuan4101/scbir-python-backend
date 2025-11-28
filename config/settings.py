import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    IMAGE_SIZE_WIDTH: int = int(os.getenv("IMAGE_SIZE_WIDTH", "256"))
    IMAGE_SIZE_HEIGHT: int = int(os.getenv("IMAGE_SIZE_HEIGHT", "256"))
    
    @property
    def IMAGE_SIZE(self) -> tuple:
        return (self.IMAGE_SIZE_WIDTH, self.IMAGE_SIZE_HEIGHT)
    
    def validate(self):
        if not self.SUPABASE_URL:
            raise ValueError("❌ Falta variable de entorno: SUPABASE_URL")
        if not self.SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("❌ Falta variable de entorno: SUPABASE_SERVICE_ROLE_KEY")
        print("✅ Configuración validada correctamente")

settings = Settings()
settings.validate()
