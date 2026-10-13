"""
End-to-end pipeline: intent -> tone -> retrieve -> answerability -> reply (or decline).
Used by app.py, the tests and the evaluation scripts.
"""
import logging

from src.answerability import check_answerability
from src.intent_classifier import load_classifier, predict_intent
from src.kb_retriever import DEFAULT_INDEX_PATH, DEFAULT_KB_PATH, KnowledgeBase
from src.reply import compose_reply
from src.tone import score_tone

log = logging.getLogger(__name__)


def _load_knowledge_base() -> KnowledgeBase:
    """Load the saved index; rebuild it if it is missing or older than the FAQ file."""
    if DEFAULT_INDEX_PATH.exists():
        kb = KnowledgeBase.load()
        if not kb.is_stale(DEFAULT_KB_PATH):
            return kb
        log.warning("FAQ file changed since the index was built; rebuilding the index.")
    else:
        log.warning("No saved index found; building one from the FAQ file.")
    kb = KnowledgeBase.build(DEFAULT_KB_PATH)
    kb.save()
    return kb


class SupportAssistant:
    def __init__(self):
        self.classifier = load_classifier()
        self.kb = _load_knowledge_base()

    def handle(self, query: str, top_k: int = 3) -> dict:
        query = query if isinstance(query, str) else ""
        intent = predict_intent(self.classifier, query)
        tone = score_tone(query)
        retrieved = self.kb.search(query, top_k=top_k)

        verdict = check_answerability(query, retrieved)
        if verdict["abstain"]:
            answer, backend, sources = verdict["message"], "abstain-rule", []
        else:
            reply = compose_reply(retrieved, tone["label"])
            answer, backend, sources = reply["answer"], reply["backend"], reply["sources"]

        return {
            "query": query,
            "intent": intent,
            "sentiment": tone,
            "retrieved": retrieved,
            "answer": answer,
            "generation_backend": backend,
            "sources": sources,
            "abstained": verdict["abstain"],
            "abstain_reason": verdict["reason"],
        }


if __name__ == "__main__":
    assistant = SupportAssistant()
    for q in ["How long does shipping usually take?", "Where is my order #55123?"]:
        r = assistant.handle(q)
        print("Q:", q, "\nA:", r["answer"], "\nSources:", r["sources"], "\n")