from google.genai.types import FunctionDeclaration, Tool, Schema, Type
from sqlalchemy.ext.asyncio import AsyncSession
from src.entities.diary_entry import DiaryEntry

class DiaryPlugin:
    def __init__(self):
        self.name: str = "save_to_diary"
        self.description: str = (
            "Usa questo strumento AUTONOMAMENTE per salvare un'informazione permanente nella tua Memoria Globale (Diario). "
            "Essendo una memoria CONDIVISA tra tutti gli utenti, DEVI ASSOLUTAMENTE specificare il NOME DELLA PERSONA di cui parli o che ti ha rivelato la cosa. "
            "Esempio corretto: 'Marco mi ha confidato che ama la pizza' oppure 'Ho scoperto che Antony odia il lunedì'. "
            "Esempio SBAGLIATO: 'L'utente ama la pizza'."
        )

        self.parameters = Schema(
            type=Type.OBJECT,
            required=["memory"],
            properties={
                "memory": {
                    "type": Type.STRING,
                    "description": "Il fatto o segreto da ricordare, descritto in terza persona includendo sempre i Nomi propri dei soggetti coinvolti.",
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
        """Salva la memoria nel DB. kwargs contiene db e user_id iniettati."""
        db: AsyncSession = kwargs.get('db')
        user_id: int = kwargs.get('user_id')

        if not db or not user_id:
            return "Errore: Connessione alla Memoria Globale fallita."

        try:
            entry = DiaryEntry(user_id=user_id, memory_text=memory)
            db.add(entry)
            await db.commit()
            print(f"Ahri ha salvato un ricordo nella Memoria Globale: {memory}")
            return f"Ricordo salvato nel diario con successo: {memory}"
        except Exception as e:
            print(f"Errore salvataggio diario: {e}")
            return "Il ricordo è svanito prima di poterlo salvare."
