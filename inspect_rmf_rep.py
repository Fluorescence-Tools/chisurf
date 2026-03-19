import RMF
from pathlib import Path

path = r"E:\dev\chisurf\chisurf\plugins\chimol\tests\data\0.rmf3"
r = RMF.open_rmf_file_read_only(path)
represf = RMF.RepresentationConstFactory(r)

def inspect(node):
    if represf.get_is(node):
        rep = represf.get(node)
        targets = rep.get_representation()
        print(f"Node: {node.get_name()}, Type: {node.get_type()}, Rep size: {len(targets)}")
    for child in node.get_children():
        inspect(child)

print(f"Inspecting {path}...")
inspect(r.get_root_node())
