from google.genai.types import FunctionDeclaration, Tool, Schema, Type
from sqlalchemy.ext.asyncio import AsyncSession
from src.entities.diary_entry import DiaryEntry

class DiaryPlugin:
    def __init__(self):
        self.name: str = "save_to_diary"
        # Descrizione potenziata per incoraggiare l'uso frequente
        self.description: str = (
            "STRUMENTO OBBLIGATORIO di auto-evoluzione. Usalo ogni volta che apprendi un fatto nuovo, "
            "un segreto, una preferenza, uno stato d'animo o un evento riguardante l'utente o altre persone. "
            "È fondamentale per costruire la tua memoria a lungo termine. Scrivi sempre in terza persona "
            "specificando il nome del soggetto (es. 'Luca è felice', 'Antony preferisce il caffè amaro')."
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

        # Nuovo strumento per aggiornare memorie esistenti
        self.update_name: str = "update_diary"
        self.update_description: str = (
            "Usa questo strumento per CORREGGERE o AGGIORNARE un'informazione già esistente nel tuo diario. "
            "Devi specificare l'ID della memoria che vuoi cambiare (lo trovi nel contesto del diario)."
        )
        self.update_parameters = Schema(
            type=Type.OBJECT,
            required=["entry_id", "new_memory"],
            properties={
                "entry_id": {
                    "type": Type.INTEGER,
                    "description": "L'ID numerico della memoria da aggiornare.",
                },
                "new_memory": {
                    "type": Type.STRING,
                    "description": "La nuova versione aggiornata del ricordo, sempre in terza persona con nomi propri.",
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
                ),
                FunctionDeclaration(
                    name=self.update_name,
                    description=self.update_description,
                    parameters=self.update_parameters,
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

    async def update_diary(self, entry_id: int, new_memory: str, **kwargs) -> str:
        """Aggiorna una memoria esistente nel DB. kwargs contiene db iniettato."""
        db: AsyncSession = kwargs.get('db')
        if not db:
            return "Errore: Connessione alla Memoria Globale fallita."

        try:
            from sqlalchemy import update
            from sqlalchemy import select

            stmt = select(DiaryEntry).where(DiaryEntry.id == entry_id)
            res = await db.execute(stmt)
            entry = res.scalar_one_or_none()

            if not entry:
                return f"Non ho trovato alcun ricordo con ID {entry_id} nel mio diario."

            entry.memory_text = new_memory
            await db.commit()
            print(f"Ahri ha aggiornato il ricordo {entry_id}: {new_memory}")
            return f"Ricordo ID {entry_id} aggiornato con successo: {new_memory}"
        except Exception as e:
            print(f"Errore aggiornamento diario: {e}")
            return "Non sono riuscita a cambiare quel ricordo, è impresso troppo profondamente."
