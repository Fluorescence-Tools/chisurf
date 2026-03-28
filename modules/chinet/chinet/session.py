import json
from .db import DB
from .base import BaseObject
from .node import Node
from .port import Port

class Session(BaseObject):
    """
    Enhanced Session class for managing a graph of Nodes and Ports.
    Mirrors the C++ Session class logic 1:1.
    """
    def __init__(self, nodes=None):
        super().__init__(name="session")
        self.nodes = {}
        if nodes:
            for name, node in nodes.items():
                self.add_node(name, node)
        self._document["type"] = "session"
        DB.register(self)

    def add_node(self, name, node):
        self.nodes[name] = node
        DB.register(node)
        for p in node.get_ports().values():
            DB.register(p)

    def get_nodes(self):
        return self.nodes

    def write_to_db(self):
        res = super().write_to_db()
        for node in self.nodes.values():
            if not node.is_connected_to_db:
                self.connect_object_to_db(node)
            res &= node.write_to_db()
        return res

    def read_from_db(self, oid):
        self.oid = oid
        # Try to recover state from global DB if already present
        stored = DB.get(oid)
        if stored:
            if isinstance(stored, Session):
                self.nodes = stored.nodes.copy()
                self.set_document(stored._document)
                return True
            elif isinstance(stored, dict):
                # It's a document from DB registry
                self.set_document(stored)
                node_oids = self._document.get("nodes", {})
                self.nodes.clear()
                for name, n_oid in node_oids.items():
                    node = DB.get(n_oid)
                    if node: self.nodes[name] = node
                return True
        
        # Fallback to base read_from_db
        res = super().read_from_db(oid)
        if res:
            node_oids = self._document.get("nodes", {})
            self.nodes.clear()
            for name, n_oid in node_oids.items():
                node = DB.get(n_oid)
                if node: self.nodes[name] = node
            return True
        return False

    def create_port(self, port_template, port_key):
        if isinstance(port_template, str): port_template = json.loads(port_template)
        p = Port(name=port_key, **port_template)
        if "value" in port_template: p.value = port_template["value"]
        DB.register(p)
        return p

    def create_node(self, node_template, node_key):
        if isinstance(node_template, str): node_template = json.loads(node_template)
        n = Node(name=node_key)
        n.set_callback(node_template.get("callback", ""), node_template.get("callback_type", ""))
        self.add_node(node_key, n)
        return n

    def read_session_template(self, json_string): return True
    def get_session_template(self): return json.dumps(self.to_dict())

    def link_nodes(self, node_name, port_name, target_node_name, target_port_name):
        n1 = self.nodes.get(node_name); n2 = self.nodes.get(target_node_name)
        if n1 and n2:
            p1 = n1.get_port(port_name); p2 = n2.get_port(target_port_name)
            if p1 and p2:
                p2.set_link(p1)
                return True
        return False

    def update(self):
        for node in self.nodes.values(): node.update()

    def to_dict(self):
        objs = DB.dump_all()
        doc = self._document.copy()
        doc["nodes"] = {k: v.oid for k, v in self.nodes.items()}
        return {"session": doc, "objects": objs}

    def save(self, filename):
        """Save session and all registered objects to a JSONL file."""
        with open(filename, 'w') as f:
            # 1. Write session object first (header)
            doc = self._document.copy()
            doc["nodes"] = {k: v.oid for k, v in self.nodes.items()}
            f.write(json.dumps(doc) + "\n")
            
            # 2. Write all other objects currently in DB
            for obj_data in DB.dump_all():
                if obj_data.get("_id") != self.oid:
                    f.write(json.dumps(obj_data) + "\n")

    def append_object(self, obj, filename):
        """Append a single object's state to an existing JSONL session file."""
        with open(filename, 'a') as f:
            if hasattr(obj, "to_dict"):
                data = obj.to_dict()
            elif hasattr(obj, "_document"):
                data = obj._document
            else:
                data = obj
            f.write(json.dumps(data) + "\n")

    @classmethod
    def load(cls, filename):
        """Load session and reconstruct object graph from a JSONL file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        if not lines: return None
        
        # Check if it's legacy monolithic JSON or new JSONL
        try:
            first_line = json.loads(lines[0])
            if not isinstance(first_line, dict): raise ValueError()
            is_jsonl = len(lines) > 1 or "session" not in first_line
        except:
            # Might be old format with one big JSON object
            with open(filename, 'r') as f: data = json.load(f)
            return cls.from_dict(data)

        if not is_jsonl:
            with open(filename, 'r') as f: data = json.load(f)
            return cls.from_dict(data)

        DB.clear()
        id_map = {}
        objects_data = []
        session = None
        
        # Pass 1: Register all raw documents and identify session
        for line in lines:
            if not line.strip(): continue
            data = json.loads(line)
            objects_data.append(data)
            DB.register(data)
            
            if data.get("type") == "session" and session is None:
                session = cls()
                session.set_document(data)
                id_map[session.oid] = session
                DB.register(session)

        # Pass 2: Instantiate Nodes and Ports
        for data in objects_data:
            oid = data.get("_id")
            if not oid or oid in id_map: continue
            
            obj_type = data.get("type")
            name = data.get("name", "")
            
            if obj_type == "port":
                p = Port(name=name, oid=oid)
                p.set_document(data)
                id_map[oid] = p; DB.register(p)
            elif obj_type == "node":
                n = Node(name=name)
                n.oid = oid; n.set_document(data)
                id_map[oid] = n; DB.register(n)

        # Pass 3: Restore Links and Connections
        for data in objects_data:
            obj = id_map.get(data.get("_id"))
            if not obj: continue
            
            if data.get("type") == "port":
                link_oid = data.get("link")
                if link_oid in id_map:
                    obj.set_link(id_map[link_oid])
            elif data.get("type") == "node":
                ports_map = data.get("ports", {})
                for pname, p_oid in ports_map.items():
                    p = id_map.get(p_oid)
                    if p:
                        obj.ports[pname] = p
                        p.set_node(obj)
                obj.fill_input_output_port_lookups()

        # Pass 4: Restore Session's top-level Node map
        if session:
            node_oids = session._document.get("nodes", {})
            for name, n_oid in node_oids.items():
                if n_oid in id_map:
                    session.nodes[name] = id_map[n_oid]

        return session

    @classmethod
    def from_dict(cls, data):
        """Compatibility shim for old monolithic JSON format."""
        DB.clear()
        sess_data = data["session"]; session = cls()
        session.set_document(sess_data); DB.register(session)
        
        for obj_data in data["objects"]:
            DB.register(obj_data)
            
        id_map = {session.oid: session}
        for obj_data in data["objects"]:
            oid = obj_data["_id"]; name = obj_data["name"]; obj_type = obj_data["type"]
            if obj_type == "port":
                p = Port(name=name, oid=oid); p.set_document(obj_data)
                id_map[oid] = p; DB.register(p)
            elif obj_type == "node":
                n = Node(name=name); n.oid = oid; n.set_document(obj_data)
                id_map[oid] = n; DB.register(n)
        
        for obj_data in data["objects"]:
            obj = id_map.get(obj_data["_id"])
            if not obj: continue
            if obj_data["type"] == "port":
                link_oid = obj_data.get("link")
                if link_oid in id_map: obj.set_link(id_map[link_oid])
            elif obj_data["type"] == "node":
                ports_map = obj_data.get("ports", {})
                for pname, p_oid in ports_map.items():
                    p = id_map.get(p_oid)
                    if p: obj.ports[pname] = p; p.set_node(obj)
                obj.fill_input_output_port_lookups()
        
        node_oids = sess_data.get("nodes", {})
        for name, n_oid in node_oids.items():
            if n_oid in id_map: session.nodes[name] = id_map[n_oid]
            
        return session

    def _update_doc_from_data(self, doc):
        doc["nodes"] = {k: v.oid for k, v in self.nodes.items()}
