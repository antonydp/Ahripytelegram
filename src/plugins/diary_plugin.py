from google.genai.types import FunctionDeclaration, Tool, Schema, Type
from sqlalchemy.ext.asyncio import AsyncSession
from src.entities.diary_entry import DiaryEntry

class DiaryPlugin:
    def __init__(self):
        self.name: str = "save_to_diary"
        self.description: str = (
            "Usa questo strumento SOLO quando l'utente ti rivela un'informazione "
            "importante su di sé (preferenze, segreti, eventi passati, legami) "
            "che vuoi ricordare per sempre. NON usarlo per chiacchiere."
        )

        self.parameters = Schema(
            type=Type.OBJECT,
            required=["memory"],
            properties={
                "memory": {
                    "type": Type.STRING,
                    "description": "L'informazione da salvare in terza persona. Es: 'L'utente ama il sushi e odia il caldo'.",
                }
            }
        )

    def get_tool(self) -> Tool:
        return Tool(
            function_declarations=[
                FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=self.parameters,
                )
            ]
        )

    async def save_to_diary(self, memory: str, **kwargs) -> str:
        """Salva la memoria nel DB. kwargs contiene db e user_id iniettati da noi."""
        db: AsyncSession = kwargs.get('db')
        user_id: int = kwargs.get('user_id')

        if not db or not user_id:
            return "Errore: Connessione al diario fallita."

        try:
            entry = DiaryEntry(user_id=user_id, memory_text=memory)
            db.add(entry)
            await db.commit()
            print(f"Ahri ha salvato un ricordo: {memory}")
            return f"Ricordo salvato nel diario con successo: {memory}"
        except Exception as e:
            print(f"Errore salvataggio diario: {e}")
            return "Il ricordo è svanito prima di poterlo salvare."
