from google.genai.types import FunctionDeclaration, Tool, Schema, Type
from ..memory_manager import ahri_memory

class MemoryPlugin:
    def __init__(self, user_id: str):
        self.__user_id = user_id
        self.__name: str = "update_diary"
        self.__description: str = "Aggiorna o aggiunge un ricordo nel diario personale riguardante l'utente corrente."

        self.__parameters = Schema(
            type=Type.OBJECT,
            required=["memory"],
            properties={
                "memory": {
                    "type": Type.STRING,
                    "description": "Il contenuto del ricordo da salvare o aggiornare.",
                }
            }
        )

    def function_declaration(self):
        return FunctionDeclaration(
            name=self.__name,
            description=self.__description,
            parameters=self.__parameters,
        )

    def get_tool(self) -> Tool:
        return Tool(
            function_declarations=[self.function_declaration()]
        )

    async def update_diary(self, memory: str) -> str:
        if not ahri_memory:
            return "Errore: Memoria non disponibile al momento."

        try:
            ahri_memory.add(
                memory,
                user_id=self.__user_id
            )
            return "Ricordo salvato correttamente nel diario."
        except Exception as e:
            print(f"Error in update_diary: {e}")
            return f"Errore durante il salvataggio del ricordo: {e}"
