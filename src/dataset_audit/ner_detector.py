"""Count person-name entities with spaCy NER."""
import spacy


def split_text(text, max_chars):
    """Split text into pieces of at most max_chars, preferring whitespace boundaries."""
    pieces = []
    start = 0
    while len(text) - start > max_chars:
        end = start + max_chars
        boundary = max(text.rfind(ws, start + 1, end) for ws in (" ", "\n", "\t"))
        if boundary > start:
            end = boundary
        pieces.append(text[start:end])
        start = end
    pieces.append(text[start:])
    return pieces


class PersonNameDetector:
    def __init__(self, model, labels=("PERSON",), batch_size=256, n_process=1, max_chunk_chars=1_000_000):
        self.nlp = spacy.load(model)
        if "ner" not in self.nlp.pipe_names:
            raise ValueError(f"spaCy model {model!r} has no NER component")
        self.nlp.select_pipes(enable=self._components_needed_by_ner())
        self.labels = set(labels)
        self.batch_size = batch_size
        self.n_process = n_process
        self.max_chunk_chars = max_chunk_chars

    def _components_needed_by_ner(self):
        needed = ["ner"]
        for name in ("tok2vec", "transformer"):
            if name in self.nlp.pipe_names and "ner" in self.nlp.get_pipe(name).listening_components:
                needed.append(name)
        return needed

    @property
    def model_info(self):
        meta = self.nlp.meta
        return {
            "name": f"{meta['lang']}_{meta['name']}",
            "version": meta["version"],
            "spacy_version": spacy.__version__,
            "components_enabled": self.nlp.pipe_names,
            "person_labels": sorted(self.labels),
        }

    def iter_counts(self, items):
        """Yield (context, entity count) in input order for (text, context) items."""
        for context, entities in self.iter_entities(items):
            yield context, len(entities)

    def iter_entities(self, items):
        """Yield (context, [entity strings]) in input order for (text, context) items."""

        def chunks():
            for text, context in items:
                pieces = split_text(text, self.max_chunk_chars)
                for i, piece in enumerate(pieces):
                    is_last = i == len(pieces) - 1
                    yield piece, (is_last, context if is_last else None)

        running = []
        docs = self.nlp.pipe(chunks(), as_tuples=True, batch_size=self.batch_size, n_process=self.n_process)
        for doc, (is_last, context) in docs:
            running.extend(ent.text for ent in doc.ents if ent.label_ in self.labels)
            if is_last:
                yield context, running
                running = []
