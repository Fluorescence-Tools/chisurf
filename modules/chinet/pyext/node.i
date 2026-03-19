/* turn on director wrapping for Callback */
%feature("director") NodeCallback;
%{
#include "../include/CNode.h"
#include "../include/DatabaseObject.h"
#include "../include/Port.h"
#include "../include/NodeCallback.h"
%}

%shared_ptr(Port)
%shared_ptr(Node)
%shared_ptr(NodeCallback)
%shared_ptr(Link)

%template(MapStringPortSharedPtr) std::map<std::string, std::shared_ptr<Port>>;
%template(ListNodePtr) std::list<std::shared_ptr<Node>>;
%template(MapStringDouble) std::map<std::string, double>;
%include "../include/NodeCallback.h"

%attribute(Node, bool, is_valid, is_valid);
%include "../include/CNode.h"

%extend Node {

    std::string __repr__(){
        std::ostringstream os;
        os << "Node(";
        const auto &in_ports = $self->get_input_ports();
        for (const auto &v : in_ports){
            os << (v.first)<< ",";
        }
        os << "->";
        const auto &out_ports = $self->get_output_ports();
        for (const auto &v : out_ports){
            os << (v.first)<< ",";
        }
        os << ")";
        return os.str();
    }

    std::shared_ptr<Port> get_port(const std::string &port_name) {
        return $self->get_port(port_name);
    }

    std::shared_ptr<Port> get_input_port(const std::string &port_name) {
        return $self->get_input_port(port_name);
    }

    std::shared_ptr<Port> get_output_port(const std::string &port_name) {
        return $self->get_output_port(port_name);
    }

    %pythoncode "node_extension.py"
}
