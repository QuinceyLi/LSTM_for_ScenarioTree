import json
from treelib import Tree

def tree_to_dict(tree, nid):
    node = tree[nid]
    if nid in [leaf.identifier for leaf in tree.leaves()]:
        return {node.tag: {}}
    else:
        tree_dict = {node.tag: {}}
        children = tree.children(nid)
        for child in children:
            tree_dict[node.tag].update(tree_to_dict(tree, child.identifier))
        return tree_dict

# 创建一个树
tree = Tree()
tree.create_node("Root", "root")
tree.create_node("Child1", "child1", parent="root")
tree.create_node("Child2", "child2", parent="root")

# 将树转换为字典
tree_dict = tree_to_dict(tree, tree.root)

# 将字典保存为JSON
with open('tree.json', 'w') as f:
    json.dump(tree_dict, f)

