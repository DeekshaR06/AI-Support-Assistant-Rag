"""
End-to-end pipeline: classify intent -> score sentiment -> retrieve relevant FAQ
articles -> check answerability -> generate a final answer (or abstain).
This is the module both app.py (Streamlit UI) and tests/test_pipeline.py call.
"""
from pathlib import Path
from src.intent_classifier import load_classifier, predict_intent
from src.sentiment import analyze_sentiment
from src.retriever import load_index, retrieve
from src.generator import generate_answer
from src.answerability import check_answerability

ROOT = Path(__file__).parent.parent


class SupportAssistant:
    def __init__(self):
        self.classifier = load_classifier()
        self.retriever_resources = load_index()

    def handle(self, query: str, top_k: int = 3) -> dict:
        intent_result = predict_intent(self.classifier, query)
        sentiment_result = analyze_sentiment(query)
        retrieved = retrieve(query, self.retriever_resources, top_k=top_k)

        verdict = check_answerability(query, retrieved)
        if verdict["abstain"]:
            answer = verdict["message"]
            backend = "abstain-rule"
        else:
            generation = generate_answer(query, retrieved, sentiment_result["label"])
            answer = generation["answer"]
            backend = generation["backend"]

        return {
            "query": query,
            "intent": intent_result,
            "sentiment": sentiment_result,
            "retrieved": retrieved,
            "answer": answer,
            "generation_backend": backend,
            "abstained": verdict["abstain"],
            "abstain_reason": verdict["reason"],
        }


if __name__ == "__main__":
    assistant = SupportAssistant()
    for q in [
        "I was charged twice for my subscription and nobody has refunded me!",
        "How long does shipping usually take?",
        "The app crashes every time I open it, so annoying",
    ]:
        result = assistant.handle(q)
        print("=" * 70)
        print("Q:", result["query"])
        print(f"Intent: {result['intent']['intent']} ({result['intent']['confidence']})")
        print(f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['compound']})")
        print(f"Top match: {result['retrieved'][0]['question'] if result['retrieved'] else 'none'}")
        print(f"Answer [{result['generation_backend']}]: {result['answer']}")