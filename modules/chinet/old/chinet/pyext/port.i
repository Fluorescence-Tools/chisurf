%{
#include "../include/Port.h"
#include "../include/CNode.h"
#ifdef WITH_MONGODB
#include "../include/MongoObject.h"
#else
#include "../include/MemoryObject.h"
#endif
%}

#ifdef WITH_MONGODB
%import "../include/MongoObject.h"
#else
%import "../include/MemoryObject.h"
%shared_ptr(MemoryObject)
#endif

%shared_ptr(Port)
%shared_ptr(Node)

%apply (double* INPLACE_ARRAY1, int DIM1)           {(double *input, int n_input)}
%apply (double** ARGOUTVIEW_ARRAY1, int* DIM1)     {(double **output, int *n_output)}
%apply (long* INPLACE_ARRAY1, int DIM1)             {(long *input, int n_input)}
%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1)        {(long **output, int *n_output)}
%apply (unsigned char* INPLACE_ARRAY1, int DIM1)    {(unsigned char *input, int n_input)}
%apply (unsigned char** ARGOUTVIEW_ARRAY1, int* DIM1){(unsigned char **output, int *n_output)}

%template(vector_port_ptr) std::vector<Port*>;

%attribute(Port, bool, fixed,     is_fixed,    set_fixed);
%attribute(Port, bool, is_output, is_output,   set_port_type);
%attribute(Port, bool, reactive,  is_reactive, set_reactive);
%attribute(Port, bool, is_linked, is_linked);
%attribute(Port, bool, bounded,   is_bounded,  set_bounded);
%attribute(Port, Node*, node,     get_node,    set_node);
%attributestring(Port, std::string, oid,  get_own_oid, set_own_oid);
%attributestring(Port, std::string, name, get_name,    set_name);

%include "../include/Port.h"

%extend Port {
    public:

    // Forward MemoryObject methods
    std::string get_own_oid() {
        return $self->MemoryObject::get_own_oid();
    }
    void set_own_oid(std::string oid_str) {
        $self->MemoryObject::set_own_oid(oid_str);
    }
    std::string get_name() {
        return $self->MemoryObject::get_name();
    }
    void set_name(std::string name) {
        $self->MemoryObject::set_name(name);
    }

    // JSON/string helpers
    std::string get_json(int indent=0) {
        return $self->Port::get_json(indent);
    }
    std::string get_json_of_key(std::string key) {
        return $self->MemoryObject::get_json_of_key(key);
    }
    std::string get_string() {
        return $self->MemoryObject::get_string();
    }
    void set_string(std::string key, std::string str) {
        $self->MemoryObject::set_string(key, str);
    }
    void set_oid(const char* key, std::string value) {
        $self->MemoryObject::set_oid(key, value);
    }

    // Template-based MemoryObject methods
    template <typename T>
    std::vector<T> get_array(const char* key) {
        return $self->MemoryObject::get_array<T>(key);
    }
    template <typename T>
    void set_array(const char* key, std::vector<T> value) {
        $self->MemoryObject::set_array<T>(key, value);
    }
    template <typename T>
    T get_singleton(const char* key) {
        return $self->MemoryObject::get_singleton<T>(key);
    }
    template <typename T>
    void set_singleton(const char* key, T value) {
        $self->MemoryObject::set_singleton<T>(key, value);
    }
    template <typename T>
    bool connect_object_to_db(T o) {
        return $self->MemoryObject::connect_object_to_db<T>(o);
    }

    // Pythonic indexing
    std::shared_ptr<MemoryObject> __getitem__(char* key) {
        return (*self)[key];
    }

    // Database connection helpers
    bool connect_to_db(
        const std::string& uri_string,
        const std::string& db_string,
        const std::string& app_string,
        const std::string& collection_string)
    {
        return $self->MemoryObject::connect_to_db(
            uri_string,
            db_string,
            app_string,
            collection_string
        );
    }
    void disconnect_from_db() {
        $self->MemoryObject::disconnect_from_db();
    }
    bool is_connected_to_db() {
        return $self->MemoryObject::is_connected_to_db();
    }

    // Instance management
    void register_instance(std::shared_ptr<MemoryObject> obj) {
        $self->MemoryObject::register_instance(obj);
    }
    void unregister_instance(std::shared_ptr<MemoryObject> obj) {
        $self->MemoryObject::unregister_instance(obj);
    }
    bool write_to_db() {
        return $self->MemoryObject::write_to_db();
    }
    std::string create_copy_in_db() {
        return $self->MemoryObject::create_copy_in_db();
    }
    bool read_from_db(const std::string& oid_string) {
        return $self->MemoryObject::read_from_db(oid_string);
    }
    bool read_json(std::string json_string) {
        return $self->MemoryObject::read_json(json_string);
    }

    // Display and comparison
    std::string show() {
        return $self->MemoryObject::show();
    }
    static std::list<std::shared_ptr<MemoryObject>> get_instances() {
        return MemoryObject::get_instances();
    }
    bool operator==(const Port& other) {
        const MemoryObject& mo_other = static_cast<const MemoryObject&>(other);
        return $self->MemoryObject::operator==(mo_other);
    }

    // Port-specific templates
    %template(set_value_vd)      set_value_vector<double>;
    %template(get_value_vd)      get_value_vector<double>;
    %template(set_value_vi)      set_value_vector<long>;
    %template(get_value_vi)      get_value_vector<long>;
    %template(set_value_d)       set_value<double>;
    %template(get_value_d)       get_value<double>;
    %template(update_buffer_d)   update_buffer<double>;
    %template(get_value_i)       get_value<long>;
    %template(set_value_i)       set_value<long>;
    %template(update_buffer_i)   update_buffer<long>;
    %template(get_value_c)       get_value<unsigned char>;
    %template(set_value_c)       set_value<unsigned char>;
    %template(update_buffer_c)   update_buffer<unsigned char>;

    %pythoncode "port_extension.py"
}
