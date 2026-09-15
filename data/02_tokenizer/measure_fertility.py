#!/usr/bin/env python
import argparse, csv
from transformers import AutoTokenizer

MODELS = {
    "gemma4e2b": "google/gemma-4-E2B-it",
    "qwen3_1p7b": "Qwen/Qwen3-1.7B",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    toks={k:AutoTokenizer.from_pretrained(v, use_fast=True) for k,v in MODELS.items()}
    with open(args.input,"r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))

    for r in rows:
        wc=len(r["text"].split())
        r["word_count_whitespace"]=wc
        for key,tok in toks.items():
            ids=tok(r["text"], add_special_tokens=False)["input_ids"]
            r[f"{key}_token_count"]=len(ids)
            r[f"{key}_fertility"]=round(len(ids)/max(1,wc),6)

    fields=list(rows[0].keys())
    with open(args.output,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

if __name__=="__main__":
    main()
