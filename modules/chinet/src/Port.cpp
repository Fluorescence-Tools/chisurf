#include "Port.h"
#include "info.h"

std::shared_ptr<Port> Port::get_ptr() {
    return std::dynamic_pointer_cast<Port>(shared_from_this());
}

std::shared_ptr<Port> Port::operator+(std::shared_ptr<Port> v)
{
    if (is_chinet_verbose()) {
        std::clog << "ADDING PORTS" << std::endl;
    }
    int new_value_type = std::max(value_type, v->value_type);
    if (is_chinet_verbose()) {
        std::clog << "-- Value type of resulting port: " << new_value_type << std::endl;
    }
    std::string name = get_name()  + " + " + v->get_name();
    if (is_chinet_verbose()) {
        std::clog << "-- Name of resulting port: " << name << std::endl;
    }
    auto re = std::make_shared<Port>(
            false, true, true, false, 0, 0, new_value_type, name
    );
    if (is_chinet_verbose()) {
        std::clog << "-- Creating a Node associated to the resulting port." << std::endl;
    }
    auto node = std::make_shared<Node>();
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
    int new_value_type = std::max(this->get_value_type(), v->get_value_type());
    re->set_value_type(new_value_type);
    auto node = std::make_shared<Node>();
    std::string name = this->get_name()  + "*" + v->get_name();
    node->set_name(name);
    node->add_input_port(this->get_name(), get_ptr());
    node->add_input_port(v->get_name(), v);
    node->add_output_port(name, re);
    if(new_value_type == 0){
        node->set_callback("multiply_int", "C");
    } else if(new_value_type == 1){
        node->set_callback("multiply_double", "C");
    }
    re->set_node(node);
    re->set_name(name);
    re->set_port_type(true);
    node->evaluate();
    return re;
}


void Port::set_link(std::shared_ptr<Port> v) {
    if (is_chinet_verbose()) {
        std::clog << "[Port::set_link] this='" << get_name() << "' -> "
                  << (v ? v->get_name() : std::string("<null>")) << std::endl;
    }
    if (v == nullptr) {
        unlink();
        return;
    }
    unlink();
    link_ = v;
    v->linked_to_.push_back(get_ptr());
    if (!node_.expired()) update_attached_node();
}

bool Port::write_to_db() {
    if (is_chinet_verbose()) {
        std::clog << "[Port::write_to_db] port='" << get_name() << "'" << std::endl;
    }
#ifdef WITH_MONGODB
    bson_t doc = get_bson();
    return MongoObject::write_to_db(doc, 0);
#else
    return MemoryObject::write_to_db();
#endif
}

bool Port::read_from_db(const std::string &oid_string) {
    if (is_chinet_verbose()) {
        std::clog << "[Port::read_from_db] oid=" << oid_string << std::endl;
    }
#ifdef WITH_MONGODB
    bool re = MongoObject::read_from_db(oid_string);
#else
    bool re = MemoryObject::read_from_db(oid_string);
#endif
#ifdef WITH_MONGODB
    auto v = MongoObject::get_array<uint8_t>("value");
#else
    auto v = get_array<uint8_t>("value");
#endif
    buffer_ = v;
    if (is_chinet_verbose()) {
        std::clog << "[Port::read_from_db] loaded buffer size=" << buffer_.size() << std::endl;
    }
    return re;
}

void Port::set_document(json doc) {
    if (is_chinet_verbose()) {
        std::clog << "[Port::set_document] keys=" << doc.size() << std::endl;
    }
    // Call the base class implementation to update the document and object fields
    MemoryObject::set_document(doc);

    // Update the Port-specific fields from the document
    if (doc.contains("fixed") && doc["fixed"].is_boolean()) {
        fixed_ = doc["fixed"].get<bool>();
    }

    if (doc.contains("is_output") && doc["is_output"].is_boolean()) {
        is_output_ = doc["is_output"].get<bool>();
    }

    if (doc.contains("is_reactive") && doc["is_reactive"].is_boolean()) {
        is_reactive_ = doc["is_reactive"].get<bool>();
    }

    if (doc.contains("is_bounded") && doc["is_bounded"].is_boolean()) {
        is_bounded_ = doc["is_bounded"].get<bool>();
    }

    if (doc.contains("value_type") && doc["value_type"].is_number()) {
        value_type = doc["value_type"].get<int>();
    }

    if (doc.contains("bounds") && doc["bounds"].is_array()) {
        bounds_.clear();
        for (auto& val : doc["bounds"]) {
            if (val.is_number()) {
                bounds_.push_back(val.get<double>());
            }
        }
    }

    // Update the buffer if the value field is present
    if (doc.contains("value") && doc["value"].is_array()) {
        if (value_type == 0) {
            // Integer values
            std::vector<long> values;
            for (auto& val : doc["value"]) {
                if (val.is_number()) {
                    values.push_back(val.get<long>());
                }
            }
            if (!values.empty()) {
                set_value_vector(values);
            }
        } else {
            // Float values
            std::vector<double> values;
            for (auto& val : doc["value"]) {
                if (val.is_number()) {
                    values.push_back(val.get<double>());
                }
            }
            if (!values.empty()) {
                set_value_vector(values);
            }
        }
    }
}

#ifndef WITH_MONGODB
std::string Port::get_json(int indent) {
    if (is_chinet_verbose()) {
        std::clog << "[Port::get_json] indent=" << indent << " (memory backend)" << std::endl;
    }
    // Update the document with the current state of the Port object
    document["fixed"] = fixed_;
    document["is_output"] = is_output_;
    document["is_reactive"] = is_reactive_;
    document["is_bounded"] = is_bounded_;
    document["value_type"] = value_type;

    // Add the value field
    if (value_type == 0) {
        long* va; int nv;
        get_value(&va, &nv);
        std::vector<long> v;
        v.assign(va, va + nv);
        json array = json::array();
        for (auto& val : v) {
            array.push_back(val);
        }
        document["value"] = array;
    } else {
        double* va; int nv;
        get_value(&va, &nv);
        std::vector<double> v;
        v.assign(va, va + nv);
        json array = json::array();
        for (auto& val : v) {
            array.push_back(val);
        }
        document["value"] = array;
    }

    // Add the bounds field
    json bounds_array = json::array();
    for (auto& val : bounds_) {
        bounds_array.push_back(val);
    }
    document["bounds"] = bounds_array;

    // Add the link field if the port is linked
    if (is_linked()) {
        document["link"] = link_->get_own_oid();
    }

    // Return the JSON representation
    return MemoryObject::get_json(indent);
}
#endif

#ifdef WITH_MONGODB
bson_t Port::get_bson()
{
    bson_t dst = get_bson_excluding("value", "bounds", NULL);
    if(value_type == 0){
        long* va; int nv;
        get_value(&va, &nv);
        auto v = std::vector<long>();
        v.assign(va, va + nv);
        append_number_array(&dst, "value", v);
    } else{
        double* va; int nv;
        get_value(&va, &nv);
        auto v = std::vector<double>();
        v.assign(va, va + nv);
        append_number_array(&dst, "value", v);
    }
    append_number_array(&dst, "bounds", bounds_);
    return dst;
}

std::string Port::get_json(int indent)
{
    if (is_chinet_verbose()) {
        std::clog << "[Port::get_json] indent=" << indent << " (mongo backend)" << std::endl;
    }
    // First get the BSON document with all the Port-specific fields
    bson_t doc = get_bson();

    // Convert the BSON document to JSON
    size_t len;
    char *str = bson_as_json(&doc, &len);
    std::string result;

    if (str) {
        if(indent == 0){
            result = std::string(str, len);
        } else{
            auto j = json::parse(str);
            result = j.dump(indent);
        }
        bson_free(str);
    } else {
        result = "{}";
    }

    // Clean up the BSON document
    bson_destroy(&doc);

    return result;
}
#endif


bool Port::bound_is_valid()
{
    bool valid = false;
    if (bounds_.size() == 2) {
        if (bounds_[0] != bounds_[1]) {
            valid = true;
        }
    }
    if (is_chinet_verbose()) {
        std::clog << "[Port::bound_is_valid] bounds.size=" << bounds_.size() << ", valid=" << std::boolalpha << valid << std::endl;
    }
    return valid;
}


void Port::set_bounds(std::vector<double> v)
{
    if (is_chinet_verbose()) {
        std::clog << "[Port::set_bounds] input.size=" << v.size() << std::endl;
    }
    if (v.size() >= 2) {
        bounds_.clear();
        double lower = std::min(v[0], v[1]);
        double upper = std::max(v[0], v[1]);
        bounds_.push_back(lower);
        bounds_.push_back(upper);
    }
}


std::vector<double> Port::get_bounds()
{
    return bounds_;
}


void Port::update_attached_node() {
    if (is_chinet_verbose()) {
        std::clog << "[Port::update_attached_node] node='" << (node_.expired() ? std::string("<expired>") : node_.lock()->get_name())
                  << "', reactive=" << std::boolalpha << is_reactive()
                  << ", is_output=" << is_output() << std::endl;
    }
    auto n = node_.lock();
    if (n == nullptr) {
        if (is_chinet_verbose()) {
            std::clog << "[Port::update_attached_node] node is null/expired, skipping update" << std::endl;
        }
        return;
    }
    n->set_valid(false);
    if (is_reactive() && !is_output()) {
        n->evaluate();
        if (is_chinet_verbose()) {
            std::clog << "[Port::update_attached_node] node evaluated" << std::endl;
        }
    }
}

// ... rest of methods ...

std::vector<std::shared_ptr<Port>> Port::get_linked_ports() {
    std::vector<std::shared_ptr<Port>> result;
    for (auto it = linked_to_.begin(); it != linked_to_.end(); ) {
        if (auto v = it->lock()) {
            result.push_back(v);
            ++it;
        } else {
            it = linked_to_.erase(it);
        }
    }
    return result;
}

// When copy=true, allocates memory using malloc() - caller is responsible for freeing.
// When used with numpy, numpy will automatically free the memory.
// When copy=false, returns a pointer to internal buffer (no allocation).
void Port::get_bytes(unsigned char **output, int *n_output, bool copy) {
    *n_output = static_cast<int>(buffer_.size());
    if (is_chinet_verbose()) {
        std::clog << "[Port::get_bytes] buffer_size=" << *n_output << ", copy=" << std::boolalpha << copy << std::endl;
    }
    if (copy) {
        auto buffer_size = *n_output;
        *output = static_cast<unsigned char*>(std::malloc(buffer_size)); // Use malloc for improved performance
        if (*output) {
            std::memcpy(*output, buffer_.data(), buffer_size);
        } else {
            throw std::runtime_error("Memory allocation failed in get_bytes.");
        }
    } else {
        *output = buffer_.data();
    }
}

void Port::set_bytes(unsigned char *input, int n_input) {
    if (is_chinet_verbose()) {
        std::clog << "[Port::set_bytes] n_input=" << n_input << std::endl;
    }
    buffer_.resize(n_input);
    std::memcpy(buffer_.data(), input, n_input);
    buffer_element_size_ = 1; // When setting raw bytes, each element is 1 byte
}

void Port::set_buffer_ptr(size_t ptr, int n_elements, int element_size) {
    if (is_chinet_verbose()) {
        std::clog << "[Port::set_buffer_ptr] n_elements=" << n_elements << ", element_size=" << element_size << std::endl;
    }
    // Validate pointer is not null
    if (ptr == 0) {
        throw std::runtime_error("Null pointer passed to set_buffer_ptr");
    }
    // Validate parameters are positive
    if (n_elements <= 0 || element_size <= 0) {
        throw std::runtime_error("Invalid parameters: n_elements and element_size must be positive");
    }
    buffer_.assign(reinterpret_cast<uint8_t*>(ptr),
                   reinterpret_cast<uint8_t*>(ptr) + (n_elements * element_size));
    buffer_element_size_ = element_size;
}

size_t Port::get_buffer_ptr() {
    if (is_chinet_verbose()) {
        std::clog << "[Port::get_buffer_ptr] returning pointer" << std::endl;
    }
    return reinterpret_cast<size_t>(buffer_.data());
}
