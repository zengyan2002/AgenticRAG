"""Benchmark validation and deterministic retrieval metrics.

Live answer evaluation is provided by knowledge.evaluation.run_quality_eval.
This module restores the public helpers used by the existing regression tests.
"""
import copy
from collections import defaultdict, deque
import math
import mimetypes
from pathlib import Path
import re
import statistics
import unicodedata


def normalize_name(value):
    return re.sub(r'[^\w]+','',unicodedata.normalize('NFKC',str(value)).casefold())


def criterion_matches(document, criterion):
    if 'any_of' in criterion:
        return any(criterion_matches(document,c) for c in criterion['any_of'])
    checks=[]
    if 'title_contains' in criterion:
        checks.append(normalize_name(criterion['title_contains']) in normalize_name(document.get('title','')))
    if 'content_contains_all' in criterion:
        content=normalize_name(document.get('content',''))
        checks.append(all(normalize_name(t) in content for t in criterion['content_contains_all']))
    return bool(checks) and all(checks)


def rank_metrics(documents, criteria, top_k=5):
    if top_k<1:
        raise ValueError('top_k 必须大于零')
    if not criteria:
        return dict.fromkeys(('recall_at_k','mrr_at_k','ndcg_at_k','first_relevant_rank'))
    hits=[[criterion_matches(d,c) for c in criteria] for d in documents]
    selected=hits[:top_k]
    recall=sum(any(row[i] for row in selected) for i in range(len(criteria)))/len(criteria)
    ranks=[i+1 for i,row in enumerate(selected) if any(row)]
    dcg=sum(1/math.log2(rank+1) for rank in ranks)
    ideal_count=min(top_k,sum(any(row) for row in hits))
    ideal=sum(1/math.log2(i+2) for i in range(ideal_count))
    return {'recall_at_k':recall,'mrr_at_k':1/ranks[0] if ranks else 0.0,
            'ndcg_at_k':dcg/ideal if ideal else 0.0,'first_relevant_rank':ranks[0] if ranks else None}


def make_stage_result(documents, criteria, top_k=5):
    return {'metrics':rank_metrics(documents,criteria,top_k),
            'candidate_metrics':rank_metrics(documents,criteria,16),'count':len(documents)}


def activate_annotation_version(case, annotation_version):
    if annotation_version not in ('v1','v2'):
        raise ValueError('未知标注版本')
    result=copy.deepcopy(case)
    for field in ('query','relevance','reference_answer','expected_query_type','expected_route_mode','expected_answerable'):
        if field in result:
            result[field+'_v1']=copy.deepcopy(case[field])
        if annotation_version=='v2' and field+'_v2' in case:
            result[field]=copy.deepcopy(case[field+'_v2'])
    result['annotation_version']=annotation_version
    return result


def validate_and_prepare_dataset(dataset, dataset_path):
    documents={d['id']:copy.deepcopy(d) for d in dataset['documents']}
    if len(documents)!=len(dataset['documents']):
        raise ValueError('文档 ID 重复')
    cases=copy.deepcopy(dataset['cases'])
    seen=set()
    for case in cases:
        if not case.get('id') or case['id'] in seen:
            raise ValueError('用例 ID 缺失或重复')
        seen.add(case['id'])
        if not str(case.get('query','')).strip():
            raise ValueError('问题不能为空')
        if case.get('expected_answerable',True):
            if case.get('document_id') not in documents or not case.get('relevance'):
                raise ValueError('可回答用例缺少文档或相关性标注')
        if case.get('judge_answer') and not str(case.get('reference_answer','')).strip():
            raise ValueError('答案评审缺少参考答案')
        if case.get('image_path'):
            path=(Path(dataset_path).parent/case['image_path']).resolve()
            if not path.is_file():
                raise ValueError(f'图片不存在: {path}')
            case['resolved_image_path']=str(path)
    return cases,documents


def load_case_image(case):
    path=Path(case.get('resolved_image_path') or case['image_path'])
    return path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream'


def determine_document_coverage(documents, available_names):
    available={normalize_name(name) for name in available_names}
    return {key:any(normalize_name(n) in available for n in [doc.get('canonical_title',''),*doc.get('aliases',[])])
            for key,doc in documents.items()}


def select_answer_judge_ids(cases, sample_size):
    groups=defaultdict(deque)
    for case in cases:
        if case.get('judge_answer'):
            groups[case.get('document_id','__unanswerable__')].append(case['id'])
    selected=[]
    size=sample_size if sample_size>0 else sum(map(len,groups.values()))
    while any(groups.values()) and len(selected)<size:
        for group in groups.values():
            if group and len(selected)<size:
                selected.append(group.popleft())
    return selected


def summarize(results, skipped_cases):
    stage_names={stage for row in results for stage in row.get('stages',{})}
    stages={}
    for stage in stage_names:
        rows=[r['stages'][stage]['metrics'] for r in results if stage in r.get('stages',{})
              and r['stages'][stage]['metrics'].get('recall_at_k') is not None]
        values={key:statistics.mean(r[key] for r in rows) if rows else None for key in ('recall_at_k','mrr_at_k','ndcg_at_k')}
        stages[stage]={'evaluated_cases':len(rows),**values}
    final=stages.get('expanded',stages.get('rerank',{}))
    return {'dataset_cases':len(results)+len(skipped_cases),'total_cases':len(results),'skipped_cases':len(skipped_cases),
            'stages':stages,'end_to_end':{key.replace('_k','_5'):final.get(key) for key in ('recall_at_k','mrr_at_k','ndcg_at_k')}}
