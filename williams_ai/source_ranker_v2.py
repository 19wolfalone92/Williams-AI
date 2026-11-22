"""Experimental audited lexical retriever, independent of exam answer labels.

No trade action, external network, model access, or private PDF content. The
ranking policy is fixed *before* measuring any historical diagnostics.
This does not establish reading comprehension or outperforming embeddings.
"""
from __future__ import annotations

from collections import Counter
from math import log
import re
import unicodedata
from typing import Any

from .book_rules import registry

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset({
    "и", "на", "по", "не", "или", "что", "это", "как", "для", "при",
    "без", "ли", "из", "об", "во", "со", "до", "бы", "уже", "есть",
    "после", "будет", "нужна", "нужно", "надо", "можно", "какая",
    "какое", "какой", "когда", "почему", "книга", "книге", "книги",
    "книжный", "книжное", "книжной", "свечи", "графика", "правило",
    "торговый", "торговой", "данные", "движение", "цена", "цене"
})

def _normalize(word: str) -> str:
    value = unicodedata.normalize("NFKC", word).casefold().replace("ё","е")
    # Suffix reduction is deliberately conservative. Both Russian and
    # English words remain in their original alphabet (no invented synonyms).
    if len(value) >= 7 and re.fullmatch(r"[а-я]+",value):
        for suffix in ("иями","ями","ами","ого","его","ому","ему",
                       "ыми","ими","иях","ов","ев","ой","ий","ый",
                       "ая","ое","ые","ых","ым","ом","ем","ам","ям",
                       "ия","ие","ах","ях","а","я","ы","и","е","у","ю"):
            if value.endswith(suffix) and len(value)-len(suffix)>=4:
                return value[:-len(suffix)]
    if len(value)>=6 and re.fullmatch(r"[a-z]+",value) and value.endswith("s"):
        return value[:-1]
    return value

def _terms(text: str) -> Counter[str]:
    tokens = (_normalize(x) for x in _WORDS.findall(text))
    return Counter(x for x in tokens if len(x)>1 and x not in _STOP)

def rank_reviewed_rules(question: str, *, edition: str, max_cards:int=12
                        ) -> list[dict[str,Any]]:
    """Review-verified rules scored with edition-local TF-IDF and field boosts.

    Searches only provided question text. Does NOT read choices, case IDs,
    expected letters, evaluation outcomes, or any model-produced prediction.
    """
    if not isinstance(question,str) or not 1<=len(question)<=1500:
        raise ValueError("INVALID_SOURCE_RANKING_QUESTION")
    if edition not in {"TC1","NTD","TC2","COURSE"}:
        raise ValueError("INVALID_SOURCE_RANKING_EDITION")
    if type(max_cards) is not int or not 1<=max_cards<=12:
        raise ValueError("INVALID_SOURCE_RANKING_LIMIT")
    query=_terms(question)
    if not query:
        return []
    cards=[c for c in registry()["rule_cards"]
           if c["source"]==edition and c["status"]=="REVIEWED_PAGES"
           and c["pdf_pages"] and len(c["summary"])<=450]
    if not cards: return []
    features=[]
    df=Counter()
    for card in cards:
        title=_terms(card["title"])
        summary=_terms(card["summary"])
        # Edition token is not a discriminative feature within an edition.
        suffix=" ".join(card["id"].split(".")[1:])
        identifier=_terms(suffix)
        features.append((card,title,summary,identifier))
        df.update(set(title)|set(summary)|set(identifier))
    n=len(cards)
    matches=[]
    for card,title,summary,identifier in features:
        score=0.0
        for token in query:
            tf=(2.0*title[token]+1.0*summary[token]+1.5*identifier[token])
            if not tf:
                continue
            # Nonnegative, length-normalized weighting; rare terms dominate
            # rather than generic "AO" / "wave" shared across many cards.
            idf=log(1+(n-df[token]+0.5)/(df[token]+0.5))
            score+=idf*(2.2*tf)/(1.2+tf)
        if score>0:
            matches.append((score,card))
    matches.sort(key=lambda r:(-r[0],r[1]["id"]))
    return [
        {"source":f'{card["source"]}: PDF {",".join(map(str,card["pdf_pages"]))}',
         "rule_id":card["id"],"status":card["status"],
         "excerpt":card["summary"],
         "confidence":"EXPERIMENTAL_SPARSE_RETRIEVAL_NOT_MODEL_VALIDATED"}
        for _,card in matches[:max_cards]
    ]
