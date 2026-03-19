#include "Session.h"
#include "Port.h"
#include "info.h"


bool Session::read_from_db(const std::string &oid_string){
    bool return_value = true;
    return_value &= DatabaseObject::read_from_db(oid_string);
#ifdef WITH_MONGODB
    return_value &= create_and_connect_objects_from_oid_array(&document, "nodes", &nodes);
#else
    return_value &= create_and_connect_objects_from_oid_array(document, "nodes", &nodes);
#endif
    return return_value;
}

bool Session::write_to_db(){

    bool re = DatabaseObject::write_to_db();

    for(auto &o : nodes){
        if(!o.second->is_connected_to_db()){
            re &= connect_object_to_db(o.second);
        }
        o.second->write_to_db();
    }

    return re;
}

#ifdef WITH_MONGODB
bson_t Session::get_bson(){
    // Since we're inheriting from DatabaseObject which is a typedef for MongoObject when WITH_MONGODB is defined,
    // we can safely cast to MongoObject* here
    bson_t doc = static_cast<MongoObject*>(this)->get_bson_excluding("nodes", NULL);
    create_oid_array_in_doc(&doc, "nodes", nodes);
    return doc;
}
#endif

std::shared_ptr<Port> Session::create_port(
        json port_template,
        std::string port_key
    ) {
    if (is_chinet_verbose()) {
        std::clog << "CREATE PORT" << std::endl;
    }
    auto port = std::make_shared<Port>();
    port->set_name(port_key);
    if (is_chinet_verbose()) {
        std::clog << "-- port name: " << port_key << std::endl;
    }
    for (json::iterator it_val = port_template.begin(); it_val != port_template.end(); ++it_val) {
        if (it_val.key() == "is_fixed") {
            bool is_fixed = port_template["is_fixed"].get<bool>();
            port->set_fixed(is_fixed);
            if (is_chinet_verbose()) {
                std::clog << "-- is_fixed: " << is_fixed << std::endl;
            }
        } else if (it_val.key() == "value") {
            auto b = port_template["value"].get<std::vector<double>>();
            if (is_chinet_verbose()) {
                std::clog << "-- number of values: " << b.size() << std::endl;
            }
            bool is_fixed = port->is_fixed();
            port->set_fixed(false);
            port->set_value(b.data(), b.size());
            port->set_fixed(is_fixed);
        } else if (it_val.key() == "is_output"){
            bool is_output = port_template["is_output"].get<bool>();
            if (is_chinet_verbose()) {
                std::clog << "-- is_output: " << is_output << std::endl;
            }
            port->set_port_type(is_output);
        } else if (it_val.key() == "is_reactive"){
            bool is_reactive = port_template["is_reactive"].get<bool>();
            if (is_chinet_verbose()) {
                std::clog << "-- is_reactive: " << is_reactive << std::endl;
            }
            port->set_reactive(is_reactive);
        }
    }
    return port;
}

std::shared_ptr<Port> Session::create_port(char* port_template, char* port_key){
    if (is_chinet_verbose()) {
        std::clog << "Session:" << port_template << ":" << port_template << std::endl;
    }
    return create_port(json::parse(port_template), port_key);
}

std::shared_ptr<Node> Session::create_node(json node_template, std::string node_key){
    if (is_chinet_verbose()) {
        std::clog << "CREATE NODE" << std::endl;
    }
    auto node = std::make_shared<Node>(node_key);
    std::string callback;
    std::string callback_type;
    for (json::iterator it = node_template.begin(); it != node_template.end(); ++it) {
        if (it.key() == "ports") {
            if (is_chinet_verbose()) {
                std::clog << "-- adding ports... " << std::endl;
            }
            auto ports_json = node_template["ports"];
            for (json::iterator it2 = ports_json.begin(); it2 != ports_json.end(); ++it2) {
                const std::string &port_key = it2.key();
                if (is_chinet_verbose()) {
                    std::clog << "-- adding port key: " << port_key << std::endl;
                }
                auto port_json = node_template["ports"][port_key];
                auto port = create_port(port_json, port_key);
                node->add_port(port_key, port, port->is_output());
            }
        } else if (it.key() == "callback") {
            callback = node_template["callback"].get<std::string>();
            if (is_chinet_verbose()) {
                std::clog << "-- callback: " << callback << std::endl;
            }
        } else if (it.key() == "callback_type") {
            callback_type = node_template["callback_type"].get<std::string>();
            if (is_chinet_verbose()) {
                std::clog << "-- callback_type: " << callback_type << std::endl;
            }
        }
    }
    node->set_callback(callback, callback_type);
    return node;
}

std::shared_ptr<Node> Session::create_node(char* node_template, char* port_key){
    return create_node(
            json::parse(node_template),
            port_key
    );
}

bool Session::read_session_template(const std::string &json_string){
    if (is_chinet_verbose()) {
        std::clog << "READ SESSION TEMPLATE" << std::endl;
    }
    json session_json = json::parse(json_string);
    // read nodes
    json nodes_json = session_json["nodes"];
    for (json::iterator it = nodes_json.begin(); it != nodes_json.end(); ++it) {
        auto node_key = it.key();
        add_node(node_key, create_node(nodes_json[node_key], node_key));
    }
    auto l = session_json["links"];
    for (json::iterator it = l.begin(); it != l.end(); ++it) {
        auto v = it.value();
        link_nodes(
                v["node"].get<std::string>(),
                v["port"].get<std::string>(),
                v["target_node"].get<std::string>(),
                v["target_port"].get<std::string>()
                );
    }
    return true;
}

bool Session::link_nodes(
        const std::string &node_name,
        const std::string &port_name,
        const std::string &target_node_name,
        const std::string &target_port_name){
    if (is_chinet_verbose()) {
        std::clog << "[Session::link_nodes] " << node_name << ":" << port_name
                  << " -> " << target_node_name << ":" << target_port_name << std::endl;
    }
    auto itn = nodes.find(node_name);
    auto itnt = nodes.find(target_node_name);

    if(itn != nodes.end() && itnt != nodes.end()){
        auto ports = nodes[node_name]->get_ports();
        auto target_ports = nodes[target_node_name]->get_ports();
        auto itp = ports.find(port_name);
        auto itpt = ports.find(target_port_name);
        if(itp != ports.end() && itpt != target_ports.end()){
            ports[port_name]->set_link(target_ports[target_port_name]);
            if (is_chinet_verbose()) {
                std::clog << "[Session::link_nodes] link created" << std::endl;
            }
            return true;
        } else if (is_chinet_verbose()) {
            std::clog << "[Session::link_nodes] ports not found" << std::endl;
        }
    } else if (is_chinet_verbose()) {
        std::clog << "[Session::link_nodes] nodes not found" << std::endl;
    }
    return false;
}

void Session::add_node(
        std::string name,
        std::shared_ptr<Node> object
)
{
    if (is_chinet_verbose()) {
        std::clog << "[Session::add_node] name=" << name << std::endl;
    }
    nodes[name] = object;
    object->set_name(name);
    if (is_connected_to_db()) {
        connect_object_to_db(object);
        if (is_chinet_verbose()) {
            std::clog << "[Session::add_node] connected node to DB" << std::endl;
        }
    }
}

std::map<std::string, std::shared_ptr<Node>> Session::get_nodes()
{
    return nodes;
}

std::string Session::get_session_template(){
    return "";
}
