#ifndef chinet_PORT_H
#define chinet_PORT_H

#include <cstdint>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "info.h"

#ifdef WITH_MONGODB
#include <bson.h>
#endif

#include "CNode.h"
#ifdef WITH_MONGODB
#include "MongoObject.h"
#else
#include "MemoryObject.h"
#endif

class Node;

#ifdef WITH_MONGODB
class Port : public MongoObject {
#else
class Port : public MemoryObject {
#endif

private:
    std::vector<uint8_t> buffer_;
    int buffer_element_size_ = 1;
    std::vector<double> bounds_{};
    std::weak_ptr<Node> node_;

    std::shared_ptr<Port> link_ = nullptr;
    std::vector<std::weak_ptr<Port>> linked_to_;

    bool remove_links_to_port() {
        if (link_ == nullptr) return false;
        auto& linked_ports = link_->linked_to_;
        auto it = std::find_if(linked_ports.begin(), linked_ports.end(), 
            [this](const std::weak_ptr<Port>& wp) {
                return wp.lock().get() == this;
            });
        if (it != linked_ports.end()) {
            linked_ports.erase(it);
            return true;
        }
        return false;
    }

    template<typename T>
    void set_value_of_dependents(const T *input, int n_input) {
        for (auto it = linked_to_.begin(); it != linked_to_.end(); ) {
            if (auto v = it->lock()) {
                v->set_value(input, n_input);
                ++it;
            } else {
                it = linked_to_.erase(it);
            }
        }
    }

    int value_type = 0;
    bool fixed_ = false;
    bool is_output_ = false;
    bool is_reactive_ = false;
    bool is_bounded_ = false;

public:
    size_t current_size() {
        return buffer_.size() / buffer_element_size_;
    }

    virtual std::shared_ptr<Port> get_ptr();

    ~Port() {
        if (is_chinet_verbose()) {
            std::clog << "DESTROYING PORT" << std::endl;
            std::clog << "-- OID: " << get_own_oid() << std::endl;
        }
        // If this Port is linked to another, remove itself from that Port's link.
        if (link_ != nullptr) {
            unlink();
        }

        // For each Port that links to this Port, remove the link.
        // This prevents other Ports from holding a dangling pointer.
        for (auto it = linked_to_.begin(); it != linked_to_.end(); ) {
            if (auto linkedPort = it->lock()) {
                if (linkedPort->is_linked() && linkedPort->get_link().get() == this) {
                    linkedPort->unlink();
                }
                ++it;
            } else {
                it = linked_to_.erase(it);
            }
        }
        linked_to_.clear();

        // MongoObject destructor will run automatically after this destructor,
        // cleaning up resources such as BSON documents and MongoDB connections.
    }

    Port(
            bool fixed = false,
            bool is_output = false,
            bool is_reactive = false,
            bool is_bounded = false,
            double lb = 0,
            double ub = 0,
            int value_type = 0,
            std::string name = ""
#ifdef WITH_MONGODB
    ) : MongoObject(name), fixed_(fixed), is_output_(is_output), is_reactive_(is_reactive), is_bounded_(is_bounded), value_type(value_type) {
#else
    ) : MemoryObject(name), fixed_(fixed), is_output_(is_output), is_reactive_(is_reactive), is_bounded_(is_bounded), value_type(value_type) {
#endif
#ifdef WITH_MONGODB
        append_string(&document, "type", "port");
#else
        document["type"] = "port";
#endif
        buffer_.resize(64); // Reserve a constant memory size for buffers
        if (is_bounded) {
            bounds_.push_back(lb);
            bounds_.push_back(ub);
        }
    }

    void set_fixed(bool fixed) { fixed_ = fixed; }
    bool is_fixed() const { return fixed_; }

    void set_port_type(bool is_output) { is_output_ = is_output; }
    bool is_output() const { return is_output_; }

    void set_reactive(bool is_reactive) { is_reactive_ = is_reactive; }
    bool is_reactive() const { return is_reactive_; }

    void set_bounded(bool is_bounded) { is_bounded_ = is_bounded; }
    bool is_bounded() const { return is_bounded_; }

    void set_value_type(int type) { 
        if (value_type == type) {
            return; // No change needed
        }

        int old_value_type = value_type;
        value_type = type;

        // If we're changing from int to float and we have existing data, convert it
        if (!buffer_.empty() && old_value_type == 0 && value_type == 1) {
            // Convert from int to float
            size_t num_elements = buffer_.size() / buffer_element_size_;
            std::vector<float> temp_buffer(num_elements);

            // Copy and convert the data
            const int* int_data = reinterpret_cast<const int*>(buffer_.data());
            for (size_t i = 0; i < num_elements; i++) {
                temp_buffer[i] = static_cast<float>(int_data[i]);
            }

            // Resize the buffer to hold the float data
            buffer_.resize(num_elements * sizeof(float));
            std::memcpy(buffer_.data(), temp_buffer.data(), num_elements * sizeof(float));
            buffer_element_size_ = sizeof(float);
        }
        // If we're changing from float to int and we have existing data, convert it
        // Note: This is a downcast and may lose precision
        else if (!buffer_.empty() && old_value_type == 1 && value_type == 0) {
            // Convert from float to int
            size_t num_elements = buffer_.size() / buffer_element_size_;
            std::vector<int> temp_buffer(num_elements);

            // Copy and convert the data
            const float* float_data = reinterpret_cast<const float*>(buffer_.data());
            for (size_t i = 0; i < num_elements; i++) {
                temp_buffer[i] = static_cast<int>(float_data[i]);
            }

            // Resize the buffer to hold the int data
            buffer_.resize(num_elements * sizeof(int));
            std::memcpy(buffer_.data(), temp_buffer.data(), num_elements * sizeof(int));
            buffer_element_size_ = sizeof(int);
        }
        // If the buffer is empty, just update the buffer_element_size_
        else if (buffer_.empty()) {
            buffer_element_size_ = (value_type == 1) ? sizeof(float) : sizeof(int);
        }
    }
    int get_value_type() const { return value_type; }

    void set_node(std::shared_ptr<Node> node_ptr) { node_ = node_ptr; }
    std::shared_ptr<Node> get_node() const { return node_.lock(); }

    void set_link(std::shared_ptr<Port> v);
    bool is_linked() const { return link_ != nullptr; }
    std::shared_ptr<Port> get_link() { return link_; }

    bool unlink() {
        if (link_ == nullptr) return false;
#ifdef WITH_MONGODB
        set_oid("link", get_bson_oid());
#else
        set_oid("link", get_own_oid());
#endif
        bool result = remove_links_to_port();
        link_ = nullptr;
        return result;
    }

    bool bound_is_valid();
    void set_bounds(std::vector<double> b);
    std::vector<double> get_bounds();

    bool is_float() {
        return ((get_value_type() == 1) || (get_value_type() == 3));
    }

    void get_bytes(unsigned char **output, int *n_output, bool copy = false);

    void set_bytes(unsigned char *input, int n_input);

    void set_buffer_ptr(size_t ptr, int n_elements, int element_size);

    size_t get_buffer_ptr();

    std::vector<std::shared_ptr<Port>> get_linked_ports();

    template<typename T>
    void set_value(const T *input, int n_input, bool copy_values = true) {
        if (is_fixed()) {
            return;
        }

        // Check if we need to change the type
        bool type_changed = false;
        int old_value_type = value_type;

        // Update value_type based on the template parameter T
        // For floating point types, set value_type to 1 (float)
        // For integer types, set value_type to 0 (int)
        // Always upcast: if current type is float and new type is int, keep it as float
        if (std::is_floating_point<T>::value) {
            // If input is float, set value_type to 1 (float)
            if (value_type != 1) {
                value_type = 1;
                type_changed = true;
            }
        } else if (value_type != 1) {
            // If input is int and current type is not float, set value_type to 0 (int)
            // This ensures we don't downcast from float to int
            value_type = 0;
        }

        // If the type has changed, update buffer_element_size_ and convert existing data if necessary
        if (type_changed) {
            if (buffer_.empty()) {
                // If the buffer is empty, just update buffer_element_size_
                buffer_element_size_ = sizeof(T);
            } else if (old_value_type == 0 && value_type == 1) {
                // Convert from int to float
                size_t num_elements = buffer_.size() / buffer_element_size_;
                std::vector<float> temp_buffer(num_elements);

                // Copy and convert the data
                const int* int_data = reinterpret_cast<const int*>(buffer_.data());
                for (size_t i = 0; i < num_elements; i++) {
                    temp_buffer[i] = static_cast<float>(int_data[i]);
                }

                // Resize the buffer to hold the float data
                buffer_.resize(num_elements * sizeof(float));
                std::memcpy(buffer_.data(), temp_buffer.data(), num_elements * sizeof(float));
                buffer_element_size_ = sizeof(float);
            }
        }

        // Now set the new values
        if (copy_values) {
            buffer_.resize(n_input * sizeof(T));
            std::memcpy(buffer_.data(), input, n_input * sizeof(T));
        } else {
            buffer_ = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(input),
                                           reinterpret_cast<const uint8_t*>(input) + n_input * sizeof(T));
        }
        buffer_element_size_ = sizeof(T);
        if (!node_.expired()) {
            update_attached_node();
            set_value_of_dependents(input, n_input);
        }
    }

    template<typename T>
    void get_value(T **output, int *n_output) {
        if (!is_linked()) {
            if (is_chinet_verbose()) {
                std::clog << "GET VALUE" << std::endl;
                std::clog << "-- Name of Port: " << get_name() << std::endl;
                std::clog << "-- Local value type: " << value_type << std::endl;
                std::clog << "-- Local buffer is filled: " << (current_size() > 0) << std::endl;
                std::clog << "-- Number of elements in local buffer: " << current_size() << std::endl;
                std::clog << "-- Port is not linked." << std::endl;
            }
            get_own_value(output, n_output);
        } else {
            if (is_chinet_verbose()) {
                std::clog << "GET VALUE" << std::endl;
                std::clog << "-- Port is linked to " << get_link()->get_name() << std::endl;
            }
            get_link()->get_value(output, n_output);
        }
        if (is_chinet_verbose()) {
            std::clog << "-- Number of elements: " << *n_output << std::endl;
        }
    }

    template<typename T>
    void get_own_value(T **output, int *n_output) {
        if (buffer_.empty()) {
            update_buffer<T>();
        }
        *n_output = buffer_.size() / buffer_element_size_;
        *output = reinterpret_cast<T*>(buffer_.data());
    }

    template<typename T>
    void set_value_vector(const std::vector<T>& input) {
        // Reuse the existing set_value method
        set_value<T>(input.data(), input.size(), true);
    }

    template<typename T>
    std::vector<T> get_value_vector() const {
        const std::vector<uint8_t>* buff = &buffer_;
        const int element_size = is_linked() ? link_->buffer_element_size_ : buffer_element_size_;
        if (is_linked()) {
            buff = &link_->buffer_;
        }
        size_t n_elements = buff->size() / element_size;
        std::vector<T> output(n_elements);

        if (!output.empty()) {
            std::memcpy(output.data(), buff->data(), n_elements * sizeof(T));
        }
        return output;
    }

#ifdef WITH_MONGODB
    virtual bson_t get_bson() final;
    virtual std::string get_json(int indent=0) override;
#endif

    template<typename T>
    void update_buffer() {
        auto v = get_array<T>("value");
        buffer_.resize(v.size() * sizeof(T));
        std::memcpy(buffer_.data(), v.data(), v.size() * sizeof(T));
        buffer_element_size_ = sizeof(T);
    }

    std::vector<uint8_t>& get_buffer() { return buffer_; }

    void update_attached_node();

    bool write_to_db();
    bool read_from_db(const std::string &oid_string);

#ifndef WITH_MONGODB
    std::string get_json(int indent=0) override;
#endif

    // Override set_document to handle the value field
    void set_document(json doc);

    std::shared_ptr<Port> operator+(std::shared_ptr<Port> v);
    std::shared_ptr<Port> operator*(std::shared_ptr<Port> v);

};

#endif //chinet_PORT_H
