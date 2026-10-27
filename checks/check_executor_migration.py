"""Check document-only executor migration. No Harness or research run occurs."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'migration'/'九阶段原执行包_原样.zip'
PREFIX='数字代理选型_九阶段执行大纲/'
with zipfile.ZipFile(ARCHIVE) as z:
    old={n[len(PREFIX):]:z.read(n) for n in z.namelist() if n.startswith(PREFIX) and not n.endswith('/')}
checks=[]
def check(name,ok,detail=''):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})

def read(name):
    return (ROOT/name).read_text(encoding='utf-8')
def between(text,a,b):
    return text.split(a,1)[1].split(b,1)[0]

main_name='数字代理选型_EXAM01-09_日程与验收大纲.md'
main=read(main_name)
old_main=old[main_name].decode()
source_rec=json.loads(read('来源与交付范围.json'))
a=json.loads(old['九阶段任务描述.json'])
b=json.loads(read('九阶段任务描述.json'))
check('task_status_unchanged',a['status']==b['status']=='PLAN_NOT_EXECUTED')
check('exactly_nine_stages',[s['id'] for s in b['stages']]==list(range(1,10)))
check('parameter_file_byte_identical',(ROOT/'验收参数_待批准.yaml').read_bytes()==old['验收参数_待批准.yaml'])
for i,(s,t) in enumerate(zip(a['stages'],b['stages']),1):
    check(f'EXAM{i:02d}_only_owner_changed',{k:v for k,v in s.items() if k!='owner'}=={k:v for k,v in t.items() if k!='owner'})
    check(f'EXAM{i:02d}_owner_harness','DeepSeek Harness' in t['owner'] and 'Codex' not in t['owner'])
    for pattern in [f'predictions/PREDICTION{i:02d}.yaml',f'templates/RESULT{i:02d}.template.json']:
        check(f'{pattern}_byte_identical',(ROOT/pattern).read_bytes()==old[pattern])
    path=f'workorders/EXAM{i:02d}.md'
    original=old[path].decode()
    expected=original.replace('**负责人**：'+s['owner']+'。','**负责人**：'+t['owner']+'。').replace('Codex','DeepSeek Harness')
    expected=expected.replace('# 工作卡（未执行）','# DeepSeek Harness工作卡（未执行）',1)
    expected=expected.replace('\n\n### EXAM','\n\n> 执行前读取同目录上一级的 `DeepSeek_Harness_执行说明.md`；本卡的预测、工作、验收和失败分支沿用原版，不授权越级执行。实体拍摄和批准仍由人工完成；独立评价不得与开发共享未揭示测试上下文。\n\n### EXAM',1)
    check(f'EXAM{i:02d}_workcard_only_executor_edits',read(path)==expected)
    check(f'EXAM{i:02d}_owner_matches_main',f"**负责人**：{t['owner']}。" in main)

check('statistics_section_exactly_preserved',between(main,'## 三、','## 四、')==between(old_main,'## 三、','## 四、'))
check('dependency_branches_exactly_preserved',between(main,'## 五、','## 六、')==between(old_main,'## 五、','## 六、'))
check('publication_mapping_exactly_preserved',between(main,'## 八、','## 九、')==between(old_main,'## 八、','## 九、'))
old_summary=old['九阶段总日程与验收一览.md'].decode()
summary=read('九阶段总日程与验收一览.md')
old_rows=[l for l in old_summary.splitlines() if l.startswith('|')]
new_rows=[l for l in summary.split('## DeepSeek Harness各阶段职责')[0].splitlines() if l.startswith('|')]
check('schedule_and_gates_summary_unchanged',old_rows==new_rows)
active=[ROOT/main_name,ROOT/'九阶段总日程与验收一览.md',ROOT/'九阶段任务描述.json',ROOT/'DeepSeek_Harness_执行说明.md',ROOT/'DeepSeek_Harness_首轮启动指令.md']+list((ROOT/'workorders').glob('*.md'))
check('active_tasks_no_codex_assignments',all(re.search('codex',p.read_text(encoding='utf-8'),re.I) is None for p in active))
check('entry_documents_exist',all((ROOT/n).is_file() for n in ['DeepSeek_Harness_执行说明.md','DeepSeek_Harness_首轮启动指令.md','执行主体迁移说明.md','README.md']))
check('unknown_cli_not_claimed_implemented','不保证符合任何Harness的原生导入schema' in read('DeepSeek_Harness_执行说明.md'))
check('human_capture_retained','人工仍负责实体设备设置、实际拍摄' in read('DeepSeek_Harness_执行说明.md'))
check('independent_context_retained','不同角色名或不同窗口不等于独立隔离' in read('DeepSeek_Harness_执行说明.md'))
check('executor_not_research_detector','不因此成为四个研究检测器之一' in read('DeepSeek_Harness_执行说明.md'))
check('first_round_is_exam01','本轮只执行EXAM01' in read('DeepSeek_Harness_首轮启动指令.md'))
check('no_runtime_execution_claim',source_rec['actual_harness_connected'] is False and source_rec['actual_harness_started'] is False and source_rec['research_experiments_run'] is False)
check('original_package_checksum',hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==source_rec['input_execution_package_sha256'])
check('no_scientific_result_directory',not (ROOT/'results').exists() and not (ROOT/'scores').exists())
# Check all active JSON/YAML parse; templates are still blank.
for p in ROOT.rglob('*'):
    if p.is_file() and 'migration' not in p.parts:
        if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'))
        elif p.suffix in {'.yaml','.yml'}:yaml.safe_load(p.read_text(encoding='utf-8'))
check('json_yaml_parseable',True)
report={'scope':'DOCUMENT_MIGRATION_CHECKS_ONLY','check_count':len(checks),'passed':sum(x['pass'] for x in checks),'failed':sum(not x['pass'] for x in checks),'checks':checks}
(ROOT/'checks'/'执行器迁移检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['scope','check_count','passed','failed']},ensure_ascii=False))
if report['failed']:
    print('\n'.join(x['name'] for x in checks if not x['pass']))
    raise SystemExit(1)
