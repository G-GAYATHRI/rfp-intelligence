# Generated demo artifacts

Run the generator from the repository root after installing the requirements,
placing the Bid1 and Bid2 source folders under `data/`, and configuring an
LLM provider and model in `.env`:

```bash
python -m scripts.generate_demo_artifacts
```

The command runs the existing extraction graph for Bid1 and Bid2, then runs ten
selected questions (five per bid) through the existing hybrid-rerank Q&A path.
It writes:

- `Bid1.json` and `Bid2.json`: extraction records using the project's canonical
  20-field schema, addendum changes, validation counts, and errors.
- `sample_qa.json`: ten Q&A records with answer confidence and source file,
  page, and chunk ID.
- `manifest.json`: generation timestamp and artifact counts.

The generator checks that an LLM provider, API key, and model are configured
before starting. It reports how many of the ten answers have traceable citations;
review that count and inspect the answers against the source PDFs before using
these files as a submission or production-quality demo.

These are runtime-generated examples, not hand-authored expected answers.
