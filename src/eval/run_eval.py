"""Run retrieval evaluation against the production Chroma/embedding pipeline."""
import argparse,json
from pathlib import Path
from src.eval.metrics import load_pairs,run_evaluation
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument("--k",type=int,default=5);p.add_argument("--pairs",default=str(ROOT/"eval/qa_questions.json"));p.add_argument("--output",default=str(ROOT/"eval/production_results.json"));a=p.parse_args()
 pairs=load_pairs(Path(a.pairs)); results=run_evaluation(pairs,["vector","hybrid","hybrid_rerank"],a.k)
 report={"evaluation_type":"production Chroma + configured SentenceTransformer embeddings + BM25 + configured CrossEncoder reranker","dataset_size":len(pairs),"results":results}
 Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
