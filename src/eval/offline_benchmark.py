"""Dependency-light retrieval benchmark over the checked-in, source-derived corpus.

The offline vector baseline is TF-IDF cosine (not the production sentence-transformer
embedding). The rerank baseline is a transparent lexical relevance reranker, not a
cross-encoder. Use `python -m src.eval.run_eval --production` after installing
requirements to benchmark the real Chroma + embedding + CrossEncoder stack.
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[2]
TOKEN_RE=re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*",re.I)

def tokens(s): return TOKEN_RE.findall(s.lower())
def load():
    chunks=[json.loads(x) for x in (ROOT/'eval/corpus_chunks.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    pairs=json.loads((ROOT/'eval/qa_questions.json').read_text(encoding='utf-8'))
    return chunks,pairs

def bm25_scores(query, docs, k1=1.5, b=.75):
    q=set(tokens(query)); corp=[tokens(d) for d in docs]; N=len(corp)
    avg=sum(map(len,corp))/max(N,1); df=Counter(t for doc in corp for t in set(doc)); out=[]
    for doc in corp:
        tf=Counter(doc); score=0.0
        for term in q:
            if term not in df: continue
            idf=math.log(1+(N-df[term]+.5)/(df[term]+.5))
            f=tf[term]; score+=idf*(f*(k1+1))/(f+k1*(1-b+b*len(doc)/(avg or 1)))
        out.append(score)
    return out

def target_match(chunk,pair):
    if chunk['bid_id']!=pair['bid_id']: return False
    if pair.get('expect_file_contains','').lower() not in chunk['file'].lower(): return False
    return all(x.lower() in chunk['text'].lower() for x in pair['expect_text_contains'])

def metrics(ranked,pairs,k=5):
    recalls=[]; rr=[]; ndcgs=[]; misses=[]
    for pair,indices in zip(pairs,ranked):
        rel=[1 if target_match(chunks[i],pair) else 0 for i in indices]
        found=next((r+1 for r,x in enumerate(rel) if x),None)
        recalls.append(1.0 if found and found<=k else 0.0); rr.append(1/found if found else 0.0)
        dcg=sum((2**v-1)/math.log2(pos+2) for pos,v in enumerate(rel[:k]))
        idcg=1.0  # one judged relevant passage per query
        ndcgs.append(dcg/idcg if idcg else 0.0)
        if not found or found>k: misses.append(pair['id'])
    n=max(len(pairs),1)
    return {'recall_at_5':round(sum(recalls)/n,4),'mrr':round(sum(rr)/n,4),'ndcg_at_5':round(sum(ndcgs)/n,4),'misses':misses}

def run():
    global chunks
    chunks,pairs=load(); docs=[c['text'] for c in chunks]
    # Dense-style lexical vector baseline, included because embedding packages may not be installed.
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words='english',max_features=100000)
    matrix=vec.fit_transform(docs)
    vector_rank=[]; hybrid_rank=[]; rerank_rank=[]
    for pair in pairs:
        q=pair['query']; allowed=[i for i,c in enumerate(chunks) if c['bid_id']==pair['bid_id']]
        qv=vec.transform([q]); sims=cosine_similarity(qv,matrix[allowed]).ravel()
        vorder=[allowed[i] for i in sims.argsort()[::-1]]
        vector_rank.append(vorder[:50])
        bm=bm25_scores(q,[docs[i] for i in allowed])
        border=[allowed[i] for i in sorted(range(len(allowed)),key=lambda x:bm[x],reverse=True)]
        # RRF over dense-style TF-IDF and BM25 ranks
        score={}
        for order in (vorder,border):
            for rank,idx in enumerate(order,1): score[idx]=score.get(idx,0)+1/(60+rank)
        fused=sorted(score,key=score.get,reverse=True)
        hybrid_rank.append(fused[:50])
        # Deterministic second-stage relevance scoring: exact identifier / phrase coverage,
        # query-term coverage and character n-gram similarity; not represented as a neural model.
        qterms=set(tokens(q))
        def rerank_score(i):
            t=chunks[i]['text'].lower(); toks=set(tokens(t))
            overlap=len(qterms&toks)/max(len(qterms),1)
            exact=sum(1 for term in qterms if len(term)>4 and term in t)
            return (exact*.15)+overlap
        rerank_rank.append(sorted(fused,key=rerank_score,reverse=True)[:50])
    results=[]
    for name,ranked in [('vector_tfidf_offline',vector_rank),('hybrid_tfidf_bm25_offline',hybrid_rank),('hybrid_plus_lexical_rerank_offline',rerank_rank)]:
        results.append({'configuration':name,**metrics(ranked,pairs,5)})
    report={'evaluation_type':'offline reproducible baseline; TF-IDF cosine + BM25 + deterministic lexical reranker; NOT production dense embeddings or CrossEncoder','dataset_size':len(pairs),'corpus_chunks':len(chunks),'corpus_pages':len({(c['bid_id'],c['file'],c['page']) for c in chunks}),'results':results}
    out=ROOT/'eval'/'offline_results.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__': run()
