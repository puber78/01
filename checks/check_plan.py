"""Validate planning documents only; this does not run research experiments."""
from pathlib import Path
import hashlib, json, re
import yaml

root = Path(__file__).resolve().parents[1]
checks=[]
def record(name, condition, detail=''):
    checks.append({'name':name,'pass':bool(condition),'detail':detail})

main_path=root/'数字代理选型_EXAM01-09_日程与验收大纲.md'
main=main_path.read_text(encoding='utf-8')
summary=(root/'九阶段总日程与验收一览.md').read_text(encoding='utf-8')
source_record=json.loads((root/'来源与交付范围.json').read_text(encoding='utf-8'))
config=yaml.safe_load((root/'验收参数_待批准.yaml').read_text(encoding='utf-8'))
tasks=json.loads((root/'九阶段任务描述.json').read_text(encoding='utf-8'))['stages']
ids=list(range(1,10))
record('nine_stage_heading_ids',re.findall(r'^### EXAM(\d{2})｜',main,re.M)==[f'{i:02d}' for i in ids])
record('nine_prediction_heading_ids',re.findall(r'^#### PREDICTION(\d{2})｜',main,re.M)==[f'{i:02d}' for i in ids])
record('nine_result_heading_ids',re.findall(r'^#### RESULT(\d{2})｜',main,re.M)==[f'{i:02d}' for i in ids])
record('nine_workorders',len(list((root/'workorders').glob('EXAM*.md')))==9)
record('nine_prediction_templates',len(list((root/'predictions').glob('PREDICTION*.yaml')))==9)
record('nine_result_templates',len(list((root/'templates').glob('RESULT*.json')))==9)
record('task_ids', [s['id'] for s in tasks]==ids)
record('dependency_order',all(all(d<s['id'] for d in s['depends']) for s in tasks))
record('early_stop_can_finalize',tasks[-1]['depends']==[] and 'early_stop_route' in config['conditional_stage9_entry'])
record('configuration_unapproved',config['approval_status']=='DRAFT_UNAPPROVED')
record('no_fixed_calendar_start',config['project_start_date'] is None)
record('inherited_source_budget',config['inherited_budget_examples']['pilot_capture_source_groups']==240 and config['inherited_budget_examples']['calibration_pool_groups']==80 and config['inherited_budget_examples']['pilot_check_groups']==160)
record('additional_digital_pool_not_invented',config['inherited_budget_examples']['digital_selection_groups'] is None)
record('confirmatory_n_not_invented', all(config['unresolved_before_confirmatory'][k] is None for k in ['n_C_confirm','n_D_sel_confirm','n_T_confirm']))
record('source_preserved',hashlib.sha256((root/'inputs'/'研究切口建议_数字代理评测与真实翻拍选型_v2_原样.md').read_bytes()).hexdigest()==source_record['input_sha256'])
original=Path(source_record['input_path'])
record('mounted_original_unchanged',not original.exists() or hashlib.sha256(original.read_bytes()).hexdigest()==source_record['input_sha256'])
record('proposal_numbers_marked','本轮建议预注册的参数' in main and '来源没有给出这些统一阈值' in main)
record('no_g9_or_raw_requirement',config['main_scope']['g9_required'] is False and config['main_scope']['raw_required'] is False)
record('fixed_and_reselected_uncertainty_distinguished','conditional_on_calibration_pool' in main and '联合、组级重采样' in main)
record('no_optional_peeking','不得反复查看p值' in main and '直到显著' in main)
record('no_pilot_as_confirmation','不能改名正式独立测试' in main)
record('record_completeness_not_success_rate','不意味着实拍成功率或预测成功率必须100%' in main)
record('same_budget_strong_baselines','S_real(b)' in main and 'S_shrink(b)' in main)
record('no_capture_cost_multiplied_by_models','捕获成本不乘模型数' in main)
record('full_pool_replays_not_independent','只改变种子不会产生新抽样证据' in main)
record('no_zero_point_five_imputation','缺失不补分数' in main and '缺失分数补为0.5' not in main)
record('new_candidate_requires_competition','至少2个未参与' in main)
record('confidence_and_practicality_separate', all(s in main for s in ['L>δ','U<−δ','L≥−ε 且 U≤ε','INCONCLUSIVE']))
record('markdown_code_fences_balanced',main.count('```')%2==0)
record('display_math_balanced',main.count('\\[')==main.count('\\]'))
required=['question','prediction','owner','inputs','work','code','result','engineering_gate','scientific_gate','next','fail','paper']
record('every_stage_complete',all(all(s.get(k) for k in required) for s in tasks))

preds=[];results=[]
for i in ids:
    p=yaml.safe_load((root/'predictions'/f'PREDICTION{i:02d}.yaml').read_text(encoding='utf-8'))
    r=json.loads((root/'templates'/f'RESULT{i:02d}.template.json').read_text(encoding='utf-8'))
    preds.append(p);results.append(r)
record('prediction_mapping',all(p['exam_id']==f'EXAM{i:02d}' and p['prediction_id']==f'PREDICTION{i:02d}' for i,p in zip(ids,preds)))
record('predictions_not_falsely_frozen',all(p['status']=='DRAFT_UNAPPROVED' and p['frozen_at'] is None and p['frozen_before_evaluation'] is False for p in preds))
record('result_mapping',all(r['result_id']==f'RESULT{i:02d}' and r['prediction_id']==f'PREDICTION{i:02d}' for i,r in zip(ids,results)))
record('results_unexecuted',all(r['execution_status']=='NOT_RUN' and r['scientific_status']=='UNTESTED' for r in results))
record('no_fabricated_results',all(all(r[k] is None for k in ['estimate','ci_low','ci_high','n_source_groups','n_captures','reviewed_by']) and r['run_ids']==[] for r in results))
ranges=[]
for s in tasks:
    match=re.search(r'D(\d+)—D(\d+)',s['window'])
    ranges.append(tuple(map(int,match.groups())))
record('workday_windows_cover_1_to_40',ranges[0][0]==1 and ranges[-1][1]==40 and all(a[1]+1==b[0] for a,b in zip(ranges,ranges[1:])))
record('summary_ids', all(f'EXAM{i:02d}' in summary and f'PREDICTION{i:02d}' in summary and f'RESULT{i:02d}' in summary for i in ids))
# Basic Markdown table structural check. No unescaped pipes occur in the tables.
def table_ok(text):
    expected=None
    for line in text.splitlines():
        if line.startswith('|'):
            count=line.count('|')
            if expected is None: expected=count
            elif expected!=count: return False
        else: expected=None
    return True
record('main_tables_well_formed',table_ok(main))
record('summary_tables_well_formed',table_ok(summary))
record('no_research_run_claim',source_record['research_experiments_run'] is False and source_record['new_web_literature_search'] is False)

report={'scope':'DOCUMENT_STATIC_CHECKS_ONLY','check_count':len(checks),'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks),'checks':checks}
(root/'checks'/'文档静态检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['scope','check_count','passed','failed']},ensure_ascii=False))
if report['failed']:
    print('\n'.join(c['name'] for c in checks if not c['pass']))
    raise SystemExit(1)
