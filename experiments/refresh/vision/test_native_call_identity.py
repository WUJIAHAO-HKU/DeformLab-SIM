"""Only host trace observation changes; native launch/topology calls identical."""
from pathlib import Path
import ast,json
HERE=Path(__file__).resolve().parent
def methods(path):
    t=ast.parse(path.read_text(encoding='utf-8'))
    return {n.name:n for c in t.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
def tree(x):return ast.dump(x,include_attributes=False)
def stripped_init(node):
    keep=[]
    for stmt in node.body:
        rendered=ast.unparse(stmt)
        if 'component_path' in rendered or 'observer_robot_surface_component_trace' in rendered or 'DEFORMABLE_COMPONENT_TRACE' in rendered or rendered.startswith('self.evidence ='):continue
        keep.append(tree(stmt))
    return keep
def test():
    a=methods(HERE/'original_component_bridge.txt');b=methods(HERE/'timing_surface_bridge.py')
    assert stripped_init(a['__init__'])==stripped_init(b['__init__'])
    for name in ['frame','snapshot','finish']:assert tree(a[name])==tree(b[name])
    old=a['after_frame'].body;new=b['after_frame'].body
    assert [tree(x) for x in old[:3]]==[tree(x) for x in new[:3]]
    assert [tree(x) for x in old[-2:]]==[tree(x) for x in new[-2:]]
    assert len(new)==5
    return dict(passed=True,checks=['topology_table_and_native_start_arguments_AST_identical','native_frame_snapshot_finish_AST_identical','per_frame_counter_rejection_and_increment_AST_identical','no_full_round_readback_or_write_in_after_frame'],scope='Native binary/kernel pinned separately; host bridge only omits diagnostic trace readback/write')
if __name__=='__main__':print(json.dumps(test(),indent=2))
