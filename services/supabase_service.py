from supabase import create_client, Client
from config.settings import settings

class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        print("✅ Supabase inicializado correctamente")
    
    def get_all_carros(self):
        response = self.client.table("carros").select("id, imagen").execute()
        return response.data or []
    
    def get_carros_with_vector(self, vector_field: str):
        response = self.client.table("carros").select(f"id, imagen, {vector_field}").execute()
        return response.data or []
    
    def update_vector(self, record_id: int, vector_field: str, vector_value: str):
        self.client.table("carros").update({
            vector_field: vector_value
        }).eq("id", record_id).execute()

supabase_service = SupabaseService()
