#include "CNode.h"
#include "Port.h"
#include "info.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <cctype>
#include <algorithm>

// Constructor
//--------------------------------------------------------------------

Node::Node(
        std::string name,
        const std::map<std::string, std::shared_ptr<Port>>& ports,
        std::shared_ptr<NodeCallback> callback_class
) : DatabaseObject(name) {
#ifdef WITH_MONGODB
    append_string(&document, "type", "node");
#else
    document["type"] = "node";
#endif

    if (is_chinet_verbose()) {
        std::clog << "[Node::ctor] Constructing node '" << name << "'\n";
        std::clog << "[Node::ctor] Ports passed: " << ports.size() << "\n";
        std::clog << "[Node::ctor] Callback provided: " << std::boolalpha << static_cast<bool>(callback_class) << std::endl;
    }

    if (!ports.empty()) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::ctor] Ports passed: " << ports.size() << "\n";
        }
        for(auto &o : ports){
            if (is_chinet_verbose()) {
                std::clog << "[Node::ctor]  -> port key='" << o.first
                          << "' is_output=" << std::boolalpha << o.second->is_output()
                          << std::endl;
            }
            o.second->set_name(o.first);
        }
        set_ports(ports);
    } else {
        if (is_chinet_verbose()) {
            std::clog << "[Node::ctor] No ports passed\n";
        }
    }

    if(callback_class != nullptr){
        if (is_chinet_verbose()) {
            std::clog << "[Node::ctor] Callback class provided\n";
        }
        this->callback_class = callback_class;
        callback_type = 1;
        if (is_chinet_verbose()) {
            std::clog << "[Node::ctor] Callback class stored; callback_type set to 1\n";
        }
    } else {
        // No callback provided; mark as non-evaluable by default
        callback_type = -1;
        callback_type_string.clear();
        callback.clear();
        if (is_chinet_verbose()) {
            std::clog << "[Node::ctor] No callback provided; callback_type set to -1 (disabled)\n";
        }
    }
    if (is_chinet_verbose()) {
        std::clog << "[Node::ctor] Node finished constructing\n";
    }
}


// Destructor
//--------------------------------------------------------------------
Node::~Node() {
    if (is_chinet_verbose()) {
        std::clog << "[Node::dtor] Destroying node '" << object_name << "'\n";
    }
}


// Methods
//--------------------------------------------------------------------

bool Node::read_from_db(const std::string &oid_string){
    if (is_chinet_verbose()) {
        std::clog << "[Node::read_from_db] READING NODE FROM DB\n";
        std::clog << "[Node::read_from_db] Requested OID: " << oid_string << std::endl;
    }
    bool return_value = true;
    bool base_ok = DatabaseObject::read_from_db(oid_string);
    return_value &= base_ok;

    if (is_chinet_verbose()) {
        std::clog << "[Node::read_from_db] Base DatabaseObject::read_from_db: "
                  << std::boolalpha << base_ok << std::endl;
    }

#ifdef WITH_MONGODB
    bool ports_ok = create_and_connect_objects_from_oid_doc(
            &document, "ports", &ports
            );
    return_value &= ports_ok;

    if (is_chinet_verbose()) {
        std::clog << "[Node::read_from_db] Ports restored: " << ports.size()
                  << " (ok=" << std::boolalpha << ports_ok << ")\n";
        std::clog << "[Node::read_from_db] callback-restore: "
                  << get_string_by_key(&document, "callback") << "\n";
        std::clog << "[Node::read_from_db] callback_type-restore: "
                  << get_string_by_key(&document, "callback_type") << std::endl;
    }

    set_callback(
            get_string_by_key(&document, "callback"),
            get_string_by_key(&document, "callback_type")
            );
#else
    bool ports_ok = create_and_connect_objects_from_oid_doc(
            document, "ports", &ports
            );
    return_value &= ports_ok;

    if (is_chinet_verbose()) {
        std::clog << "[Node::read_from_db] Ports restored: " << ports.size()
                  << " (ok=" << std::boolalpha << ports_ok << ")\n";
        std::clog << "[Node::read_from_db] callback-restore: "
                  << document["callback"].get<std::string>() << "\n";
        std::clog << "[Node::read_from_db] callback_type-restore: "
                  << document["callback_type"].get<std::string>() << std::endl;
    }

    set_callback(
            document["callback"].get<std::string>(),
            document["callback_type"].get<std::string>()
            );
#endif

    // Rebuild in/out lookups just in case
    fill_input_output_port_lookups();

    if (is_chinet_verbose()) {
        std::clog << "[Node::read_from_db] Completed with status: "
                  << std::boolalpha << return_value << std::endl;
    }
    return return_value;
}

bool Node::write_to_db() {
    if (is_chinet_verbose()) {
        std::clog << "[Node::write_to_db] Writing node '" << object_name
                  << "' with " << ports.size() << " ports\n";
    }

    bool re = DatabaseObject::write_to_db();

    if (is_chinet_verbose()) {
        std::clog << "[Node::write_to_db] Base write_to_db: "
                  << std::boolalpha << re << std::endl;
    }

    for(auto &o : ports){
        const std::string& pname = o.first;
        auto& pptr = o.second;

        if (is_chinet_verbose()) {
            std::clog << "[Node::write_to_db] Port '" << pname
                      << "' (output=" << std::boolalpha << pptr->is_output() << ") "
                      << "connected_to_db=" << pptr->is_connected_to_db() << std::endl;
        }

        if(!pptr->is_connected_to_db()){
            bool conn_ok = connect_object_to_db(pptr);
            re &= conn_ok;
            if (is_chinet_verbose()) {
                std::clog << "[Node::write_to_db]  -> connect_object_to_db: "
                          << std::boolalpha << conn_ok << std::endl;
            }
        }
        bool pw = pptr->write_to_db();
        re &= pw;

        if (is_chinet_verbose()) {
            std::clog << "[Node::write_to_db]  -> port write_to_db: "
                      << std::boolalpha << pw << std::endl;
        }
    }

    if (is_chinet_verbose()) {
        std::clog << "[Node::write_to_db] Overall result: "
                  << std::boolalpha << re << std::endl;
    }
    return re;
}

// Getter
//--------------------------------------------------------------------

std::string Node::get_name(){
    std::string r;
    r.append(object_name);
    r.append(":");
    r.append(callback);
    r.append(":");
    r.append("(");
    for(auto const &n : get_input_ports()){
        r.append(n.first);
        r.append(",");
    }
    r.append(")");

    r.append("->");

    r.append("(");
    for(auto const &n : get_output_ports()){
        r.append(n.first);
        r.append(",");
    }
    r.append(")");

    if (is_chinet_verbose()) {
        std::clog << "[Node::get_name] " << r << std::endl;
    }
    return r;
}

const std::map<std::string, std::shared_ptr<Port>>& Node::get_ports() const{
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_ports] Returning " << ports.size() << " ports\n";
    }
    return ports;
}

void Node::set_ports(const std::map<std::string, std::shared_ptr<Port>>& ports){
    if (is_chinet_verbose()) {
        std::clog << "[Node::set_ports] Setting " << ports.size() << " ports\n";
    }
    for(auto &o: ports){
        if (is_chinet_verbose()) {
            std::clog << "[Node::set_ports]  -> port key='" << o.first
                      << "' is_output=" << std::boolalpha << o.second->is_output()
                      << std::endl;
        }
        o.second->set_name(o.first);
        add_port(o.first, o.second, o.second->is_output(), false);
    }
    fill_input_output_port_lookups();
}

std::shared_ptr<Port> Node::get_port(const std::string &port_name){
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_port] Request '" << port_name << "'\n";
    }
    auto it = ports.find(port_name);
    if (it == ports.end()) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::get_port]  -> not found\n";
        }
        return nullptr;
    }
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_port]  -> found\n";
    }
    return it->second;
}

std::shared_ptr<Port> Node::get_input_port(const std::string &port_name){
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_input_port] Request '" << port_name << "'\n";
    }
    auto it = in_.find(port_name);
    if (it == in_.end()) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::get_input_port]  -> not found\n";
        }
        return nullptr;
    }
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_input_port]  -> found\n";
    }
    return it->second;
}

std::shared_ptr<Port> Node::get_output_port(const std::string &port_name){
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_output_port] Request '" << port_name << "'\n";
    }
    auto it = out_.find(port_name);
    if (it == out_.end()) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::get_output_port]  -> not found\n";
        }
        return nullptr;
    }
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_output_port]  -> found\n";
    }
    return it->second;
}

const std::map<std::string, std::shared_ptr<Port>>& Node::get_input_ports() const{
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_input_ports] Returning " << in_.size() << " inputs\n";
    }
    return in_;
}

const std::map<std::string, std::shared_ptr<Port>>& Node::get_output_ports() const{
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_output_ports] Returning " << out_.size() << " outputs\n";
    }
    return out_;
}

// Setter
//--------------------------------------------------------------------
void Node::set_callback(std::string s_callback, std::string s_callback_type){
    if (is_chinet_verbose()) {
        std::clog << "[Node::set_callback(str,str)] NODE SET CALLBACK\n";
    }
    this->callback = std::move(s_callback);
    this->callback_type_string = std::move(s_callback_type);

    // Determine numeric callback_type
    //  -1: no callback configured
    //   0: C-style callback by name
    //   1: class-based callback (see set_callback(NodeCallback))
    if (callback.empty()) {
        callback_type = -1;
    } else {
        // Normalize type string to upper-case simple form
        std::string t = callback_type_string;
        std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(::toupper(c)); });
        if (t == "C") {
            callback_type = 0;
        } else if (t == "CLASS" || t == "CPP" || t == "C++") {
            callback_type = 1;
        } else {
            callback_type = -1; // unknown type disables evaluation by default
        }
    }

    if (is_chinet_verbose()) {
        std::clog << "[Node::set_callback(str,str)] -- Callback type: " << callback_type_string << " (numeric=" << callback_type << ")\n";
        std::clog << "[Node::set_callback(str,str)] -- Callback name: " << callback << std::endl;
    }
}

void Node::set_callback(std::shared_ptr<NodeCallback> cb){
    callback_class = std::move(cb);
    callback_type = 1;
    if (is_chinet_verbose()) {
        std::clog << "[Node::set_callback(cls)] Callback class stored; callback_type=1\n";
    }
}

// Private
//--------------------------------------------------------------------

// Methods
//--------------------------------------------------------------------
void Node::add_port(
        const std::string &key,
        std::shared_ptr<Port> port,
        bool is_output,
        bool fill_in_out
        ) {
    if (is_chinet_verbose()) {
        std::clog << "[Node::add_port] ADDING PORT TO NODE\n";
        std::clog << "[Node::add_port] -- Name of node: " << get_name() << "\n";
        std::clog << "[Node::add_port] -- Key of port: " << key << "\n";
        std::clog << "[Node::add_port] -- Port is_output: " << std::boolalpha << is_output << "\n";
        std::clog << "[Node::add_port] -- Fill value of output: " << std::boolalpha << fill_in_out << std::endl;
    }
    port->set_port_type(is_output);
    port->set_node(std::dynamic_pointer_cast<Node>(shared_from_this()));
    if (ports.find(key) == ports.end() ) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::add_port] -- The key of the port was not found.\n";
            std::clog << "[Node::add_port] -- Port " << key << " was created in node.\n";
        }
        ports[key] = port;
    } else {
        auto p = ports[key];
        if(port != p){
            if (is_chinet_verbose()) {
                std::clog << "[Node::add_port] WARNING: Overwriting the port originally associated to key '" << key << "'.\n";
            }
            ports[key] = port;
        } else{
            std::cerr << "[Node::add_port] WARNING: Port is already part of the node.\n";
            std::cerr << "[Node::add_port] -- Assigning Port to the key: " << key << ".\n";
        }
    }
    if(fill_in_out){
        if (is_chinet_verbose()) {
            std::clog << "[Node::add_port] Rebuilding input/output lookups due to fill_in_out\n";
        }
        fill_input_output_port_lookups();
    }
}

void Node::add_input_port(
        const std::string &key,
        std::shared_ptr<Port> port
        ) {
    if (is_chinet_verbose()) {
        std::clog << "[Node::add_input_port] key='" << key << "'\n";
    }
    add_port(key, port, false);
}

void Node::add_output_port(
        const std::string &key,
        std::shared_ptr<Port> port
        ) {
    if (is_chinet_verbose()) {
        std::clog << "[Node::add_output_port] key='" << key << "'\n";
    }
    add_port(key, port, true);
}

#ifdef WITH_MONGODB
bson_t Node::get_bson(){
    if (is_chinet_verbose()) {
        std::clog << "[Node::get_bson] Building BSON for node '" << object_name << "'\n";
    }
    // Since we're inheriting from DatabaseObject which is a typedef for MongoObject when WITH_MONGODB is defined,
    // we can safely cast to MongoObject* here
    bson_t dst = static_cast<MongoObject*>(this)->get_bson_excluding(
            "input_ports",
            "output_ports",
             "callback",
             "callback_type",
             NULL
    );

    create_oid_dict_in_doc<Port>(&dst, "ports", ports);
    append_string(&dst, "callback", callback);
    append_string(&dst, "callback_type", callback_type_string);
    return dst;
}
#endif

void Node::evaluate(){
    if (is_chinet_verbose()) {
        std::clog << "[Node::evaluate] NODE EVALUATE\n";
        std::clog << "[Node::evaluate] -- Node name: " << get_name() << "\n";
        std::clog << "[Node::evaluate] -- Callback_type: " << callback_type << std::endl;
    }

    // If no callback is configured, do not evaluate
    if (callback_class == nullptr && callback_type < 0) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::evaluate] -- No callback configured (callback_type < 0). Skipping evaluation.\n";
        }
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (callback_class != nullptr) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::evaluate] -- Calling 'run' method of a callback class.\n";
            std::clog << "[Node::evaluate] -- Inputs: " << in_.size()
                      << ", Outputs: " << out_.size() << std::endl;
        }
        callback_class->run(in_, out_);
    } else if (is_chinet_verbose()) {
        std::clog << "[Node::evaluate] -- No callback_class set, skipping callback.\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (is_chinet_verbose()) {
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::clog << "[Node::evaluate] -- Callback duration: " << dt << " µs\n";
        std::clog << "[Node::evaluate] -- Setting nodes associated to output ports to invalid.\n";
    }

    for(auto &o : get_output_ports()){
        auto n = o.second->get_node();
        if(n != nullptr){
            if (is_chinet_verbose()) {
                std::clog << "[Node::evaluate] -- Node " << n->get_name()
                          << " of port " << o.second->get_name()
                          << " set to invalid.\n";
            }
            n->set_valid(false);
        } else if (is_chinet_verbose()) {
            std::clog << "[Node::evaluate] -- Output port '" << o.first
                      << "' has no associated node.\n";
        }
    }
    node_valid_ = true;

    if (is_chinet_verbose()) {
        std::clog << "[Node::evaluate] -- Node '" << object_name
                  << "' marked valid.\n";
    }
}

void Node::fill_input_output_port_lookups(){
    if (is_chinet_verbose()) {
        std::clog << "[Node::fill_input_output_port_lookups] Rebuilding lookups\n";
    }

    out_.clear();
    in_.clear();

    for(auto &o: ports){
        if(o.second->is_output()){
            out_[o.first] = o.second;
            if (is_chinet_verbose()) {
                std::clog << "[Node::fill_input_output_port_lookups]   out: " << o.first << "\n";
            }
        } else{
            in_[o.first] = o.second;
            if (is_chinet_verbose()) {
                std::clog << "[Node::fill_input_output_port_lookups]   in : " << o.first << "\n";
            }
        }
    }

    if (is_chinet_verbose()) {
        std::clog << "[Node::fill_input_output_port_lookups] Completed. in_="
                  << in_.size() << ", out_=" << out_.size() << std::endl;
    }
}

bool Node::inputs_valid(){
    if (is_chinet_verbose()) {
        std::clog << "[Node::inputs_valid] Checking " << in_.size() << " inputs\n";
    }

    for(const auto &i : in_){
        const auto& key = i.first;
        auto input_port = i.second;
        if(input_port->is_linked()){
            auto output_port = input_port->get_link();
            auto output_node = output_port->get_node();
            if (is_chinet_verbose()) {
                std::clog << "[Node::inputs_valid]   input '" << key << "' linked to port '"
                          << output_port->get_name() << "' in node '" << (output_node ? output_node->get_name() : "<null>") << "'\n";
            }
            if(output_node.get() == this) {
                if (is_chinet_verbose()) {
                    std::clog << "[Node::inputs_valid]   -> link is intra-node; valid\n";
                }
                return true;
            }
            else if(output_node != nullptr && !output_node->is_valid()) {
                if (is_chinet_verbose()) {
                    std::clog << "[Node::inputs_valid]   -> upstream node invalid; inputs not valid\n";
                }
                return false;
            }
        } else if (is_chinet_verbose()) {
            std::clog << "[Node::inputs_valid]   input '" << key << "' is not linked; treating as valid\n";
        }
    }
    if (is_chinet_verbose()) {
        std::clog << "[Node::inputs_valid] All inputs valid\n";
    }
    return true;
}

void Node::set_valid(bool is_valid){
    if (is_chinet_verbose()) {
        std::clog << "[Node::set_valid] Node '" << object_name
                  << "' set to " << std::boolalpha << is_valid << std::endl;
    }
    node_valid_ = is_valid;
    for(auto &v : out_)
    {
        auto output_port = v.second;
        (void)output_port; // currently unused; kept for potential future invalidation logging
        // v.second->set_invalid();
    }
}

bool Node::is_valid(){
    if(get_input_ports().empty()) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::is_valid] No inputs -> true\n";
        }
        return true;
    }
    else if(!inputs_valid()) {
        if (is_chinet_verbose()) {
            std::clog << "[Node::is_valid] Inputs not valid -> false\n";
        }
        return false;
    }
    else {
        if (is_chinet_verbose()) {
            std::clog << "[Node::is_valid] Returning node_valid_=" << std::boolalpha << node_valid_ << std::endl;
        }
        return node_valid_;
    }
}
