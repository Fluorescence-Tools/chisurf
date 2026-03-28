%{
#include "../include/Session.h"
%}

%shared_ptr(Session)
%template(MapStringNode) std::map<std::string, std::shared_ptr<Node>>;
%ignore create_port(json port_template, std::string &port_key);
%ignore create_node(json node_template, std::string &node_key);

%include "../include/Session.h"


