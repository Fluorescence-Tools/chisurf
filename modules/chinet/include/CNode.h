#ifndef chinet_Node_H
#define chinet_Node_H


#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

#include <rttr/registration>

#include "MongoObject.h"
#include "Port.h"
#include "NodeCallback.h"


// Node
//====================================================================


class Port;
class NodeCallback;
class Node : public MongoObject
{

private:
    friend Port;
    bool node_valid_ = false;
    std::map<std::string, std::shared_ptr<Port>> ports;

protected:
    void fill_input_output_port_lookups();
    rttr::method meth_ = rttr::type::get_global_method("nothing");
    std::map<std::string, std::shared_ptr<Port>> in_;
    std::map<std::string, std::shared_ptr<Port>> out_;
    std::string callback;
    std::string callback_type_string;
    int callback_type;

public:

    /// A pointer to a function that operates on an input Port instance (first argument)
    /// and writes to an output Port instance (second argument)
    std::shared_ptr<NodeCallback> callback_class;

    // Constructor & Destructor
    //--------------------------------------------------------------------
    Node(
        std::string name="",
        const std::map<std::string, std::shared_ptr<Port>>& ports = std::map<std::string, std::shared_ptr<Port>>(),
        std::shared_ptr<NodeCallback> callback_class = nullptr
    );
    ~Node();

    // Methods
    //--------------------------------------------------------------------
    bool read_from_db(const std::string &oid_string) final;
    void evaluate();
    bool is_valid();
    bool inputs_valid();
    bool write_to_db() final;

    // Getter
    //--------------------------------------------------------------------
    bson_t get_bson();

    std::string get_name();

    std::map<std::string, std::shared_ptr<Port>> get_input_ports();
    std::map<std::string, std::shared_ptr<Port>> get_output_ports();

    std::map<std::string, std::shared_ptr<Port>> get_ports();
    void set_ports(const std::map<std::string, std::shared_ptr<Port>>& ports);

    Port* get_port(const std::string &port_name);

    void add_port(
            const std::string &key,
            std::shared_ptr<Port>,
            bool is_source,
            bool fill_in_out=true
                    );
    void add_input_port(const std::string &key, std::shared_ptr<Port> port);
    void add_output_port(const std::string &key, std::shared_ptr<Port> port);

    Port* get_input_port(const std::string &port_name);
    Port* get_output_port(const std::string &port_name);

    // Setter
    //--------------------------------------------------------------------
    void set_callback(std::shared_ptr<NodeCallback> cb);
    void set_callback(std::string callback, std::string callback_type);
    void set_valid(bool is_valid=false);
};


#endif //chinet_Node_H
