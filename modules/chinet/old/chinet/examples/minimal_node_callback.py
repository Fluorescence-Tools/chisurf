"""
Minimal example: instantiate a chinet Node with a Python callback and evaluate it.

Usage:
  python examples/minimal_node_callback.py

Expected behavior:
  - Creates a Node with a simple Python callback (adds two inputs)
  - Prints input/output port names
  - Evaluates the node and prints the output value

If you're diagnosing a segfault with the MemoryObject backend, run this script
inside a clean Python environment where `chinet` is importable. This is designed
to be as small as possible while exercising the Python director callback path.
"""

import sys

try:
    import chinet as cn
except Exception as e:
    print("Failed to import chinet module:", e)
    sys.exit(1)


# Simple Python callback: add two inputs
# Defaults are provided so ports can be inferred without prior setup
# You can also annotate as strings ('float') if desired, but defaults suffice here.
def add(a=1.0, b=2.0):
    return a + b


def main():
    print("Creating Node with Python callback ...")
    # reactive_* False to avoid any background/reactive behavior; keep it minimal
    n = cn.Node(callback_function=add, reactive_inputs=False, reactive_outputs=False, name="adder")

    # Show the node representation and ports
    print("Node repr:", repr(n))
    print("Input ports:", list(n.inputs.keys()))
    print("Output ports:", list(n.outputs.keys()))

    # Evaluate and read the output
    print("Evaluating node ...")
    n.evaluate()
    # For single return value, node_extension names it 'out_00'
    out_name = "out_00"
    if out_name in n.outputs:
        print(f"Output {out_name} value:", n.outputs[out_name].value)
    else:
        # Fallback: print whatever is present
        for k, p in n.outputs.items():
            print(f"Output {k} value:", p.value)
            break


if __name__ == "__main__":
    main()
