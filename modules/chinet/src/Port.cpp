#include "Port.h"

// Operator
// --------------------------------------------------------------------
std::shared_ptr<Port> Port::operator+(
        std::shared_ptr<Port> v
        )
{
#if CHINET_VERBOSE
    std::clog << "ADDING PORTS" << std::endl;
#endif
    int new_value_type = std::max(value_type, v->value_type);
#if CHINET_VERBOSE
    std::clog << "-- Value type of resulting port: " << new_value_type << std::endl;
#endif
    std::string name = get_name()  + " + " + v->get_name();
#if CHINET_VERBOSE
    std::clog << "-- Name of resulting port: " << name << std::endl;
#endif
    auto re = std::make_shared<Port>(
            false, true, true, false, 0, 0, new_value_type, name
            );
#if CHINET_VERBOSE
    std::clog << "-- Creating a Node associated to the resulting port." << std::endl;
#endif
    auto node = new Node();
    node->set_name(name);
    node->add_input_port(this->get_name(), get_ptr());
    node->add_input_port(v->get_name(), v);
    node->add_output_port(name, re);
    if(new_value_type == 0){
        node->set_callback("addition_int", "C");
    } else if(new_value_type == 1){
        node->set_callback("addition_double", "C");
    }
    re->set_node(node);
    node->evaluate();
    return re;
}

std::shared_ptr<Port> Port::operator*(std::shared_ptr<Port> v)
{
    auto re = std::make_shared<Port>();
    re->set_value_type(this->get_value_type());
    auto node = new Node();
    std::string name = this->get_name()  + "+" + v->get_name();
    node->set_name(name);
    node->add_input_port(this->get_name(), get_ptr());
    node->add_input_port(v->get_name(), v);
    node->add_output_port(name, re);
    if(this->get_value_type() == 0){
        node->set_callback("multiply_double", "C");
    } else if(this->get_value_type() == 1){
        node->set_callback("multiply_int", "C");
    }
    re->set_node(node);
    re->set_name(name);
    re->set_port_type(true);
    node->evaluate();
    return re;
}


void Port::get_bytes(
        unsigned char **output, int *n_output,
        bool copy
        ){
    *n_output = n_buffer_elements_ * buffer_element_size_;
    if(copy){
        auto out = malloc(n_buffer_elements_ * buffer_element_size_);
        memcpy(out, buffer_, n_buffer_elements_);
        *output = static_cast<unsigned char*>(out);
    } else{
        *output = static_cast<unsigned char*>(buffer_);
    }
}

void Port::set_bytes(
        unsigned char *input, int n_input
){
    buffer_ = realloc(buffer_, n_input);
    if(buffer_ != nullptr){
        memcpy(buffer_, input, n_input);
    }
}


//
//Port Port::operator-(Port &v)
//{
//    auto a = get_value_vector<double>();
//    auto b = v.get_value_vector<double>();
//    auto result = Functions::get_vector_of_min_size(a, b);
//
//    for(int i = 0; i < result.size(); ++i){
//        result[i] = a[i] - b[i];
//    }
//
//    Port re(result);
//    re.set_name(this->get_name()  + "-" + v.get_name());
//
//    return re;
//}
//
//Port Port::operator*(Port &v)
//{
//    auto a = get_value_vector<double>();
//    auto b = v.get_value_vector<double>();
//    auto result = Functions::get_vector_of_min_size(a, b);
//
//    for(int i = 0; i < result.size(); ++i){
//        result[i] = a[i] * b[i];
//    }
//
//    Port re(result);
//    re.set_name(this->get_name()  + "*" + v.get_name());
//
//    return re;
//}
//
//
//Port Port::operator/(Port &v)
//{
//    auto a = get_value_vector<double>();
//    auto b = v.get_value_vector<double>();
//    auto result = Functions::get_vector_of_min_size(a, b);
//
//    for(int i = 0; i < result.size(); ++i){
//        result[i] = a[i] / b[i];
//    }
//
//    Port re(result);
//    re.set_name(this->get_name()  + "/" + v.get_name());
//
//    return re;
//}


// Methods
// --------------------------------------------------------------------
bool Port::read_from_db(const std::string &oid_string)
{
    bool re = MongoObject::read_from_db(oid_string);
    auto v = MongoObject::get_array<double>("value");
    n_buffer_elements_ = v.size();
    buffer_ = std::realloc(buffer_, sizeof(double) * n_buffer_elements_);
    if(buffer_ != nullptr)
    {
        n_buffer_elements_ = v.size();
        if(value_type == 0){
            memcpy(buffer_, v.data(), n_buffer_elements_ * sizeof(double));
        } else{
            memcpy(buffer_, v.data(), n_buffer_elements_ * sizeof(int));
        }
    }
    return re;
}

std::shared_ptr<Port> Port::get_ptr(){
    return std::dynamic_pointer_cast<Port>(shared_from_this());
}

bool Port::write_to_db()
{
    bson_t doc = get_bson();
    return MongoObject::write_to_db(doc, 0);
}

// Setter
//--------------------------------------------------------------------

bson_t Port::get_bson()
{
    bson_t dst = get_bson_excluding("value", "bounds", NULL);
    if(value_type == 0){
        long* va; int nv;
        get_own_value(&va, &nv);
        auto v = std::vector<long>();
        v.assign(va, va + nv);
        append_number_array(&dst, "value", v);
    } else{
        double* va; int nv;
        get_own_value(&va, &nv);
        auto v = std::vector<double>();
        v.assign(va, va + nv);
        append_number_array(&dst, "value", v);
    }
    append_number_array(&dst, "bounds", bounds_);
    return dst;
}

bool Port::bound_is_valid()
{
    if (bounds_.size() == 2) {
        if (bounds_[0] != bounds_[1]) {
            return true;
        }
    }
    return false;
}


void Port::set_bounds(double *input, int n_input)
{
    if (n_input >= 2) {
        bounds_.clear();
        double lower = std::min(input[0], input[1]);
        double upper = std::max(input[0], input[1]);
        bounds_.push_back(lower);
        bounds_.push_back(upper);
    }
}

void Port::get_bounds(double **output, int *n_output)
{
    *output = bounds_.data();
    *n_output = bounds_.size();
}




// Getter
//--------------------------------------------------------------------


bool Port::is_fixed()
{
    return get_singleton<bool>("is_fixed");
}

void Port::set_fixed(bool fixed)
{
    set_singleton("is_fixed", fixed);
}

bool Port::is_output()
{
    return get_singleton<bool>("is_output");
}

void Port::set_port_type(bool is_output)
{
    set_singleton("is_output", is_output);
}

bool Port::is_reactive()
{
    return get_singleton<bool>("is_reactive");
}

void Port::set_reactive(bool reactive)
{
    set_singleton("is_reactive", reactive);
}

bool Port::is_bounded()
{
    return get_singleton<bool>("is_bounded");
}

void Port::set_bounded(bool bounded)
{
    set_singleton("is_bounded", bounded);
}

Node* Port::get_node()
{
    return node_;
}

void Port::set_node(Node* node_ptr)
{
    node_ = node_ptr;
}

void Port::update_attached_node() {
#if CHINET_VERBOSE
    std::clog << "-- Port is reactive: " << is_reactive() << std::endl;
    std::clog << "-- Port is output: " << is_output() << std::endl;
    std::clog << "-- Updating the node: " << node_->get_name() << std::endl;
#endif
    node_->set_valid(false);
    if (is_reactive() && !is_output()) {
        node_->evaluate();
    }
#if CHINET_VERBOSE
    std::clog << "-- Node is valid: " << node_->is_valid() << std::endl;
#endif
}