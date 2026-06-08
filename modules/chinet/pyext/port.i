%{
#include "../include/Port.h"
#include "../include/CNode.h"
%}

%shared_ptr(Port)
%shared_ptr(Node)

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *input, int n_input)}
%apply (double** ARGOUTVIEW_ARRAY1, int* DIM1) {(double **output, int *n_output)}
%apply (long* INPLACE_ARRAY1, int DIM1) {(long *input, int n_input)}
%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1) {(long **output, int *n_output)}
%apply (unsigned char* INPLACE_ARRAY1, int DIM1) {(unsigned char *input, int n_input)}
%apply (unsigned char** ARGOUTVIEW_ARRAY1, int* DIM1) {(unsigned char **output, int *n_output)}

%template(vector_port_ptr) std::vector<Port*>;

%attribute(Port, bool, fixed, is_fixed, set_fixed);
%attribute(Port, bool, is_output, is_output, set_port_type);
%attribute(Port, bool, reactive, is_reactive, set_reactive);
%attribute(Port, bool, is_linked, is_linked);
%attribute(Port, bool, bounded, is_bounded, set_bounded);
%attribute(Port, Node*, node, get_node, set_node);

%include "../include/Port.h"

%extend Port{

    public:
        %template(set_value_d) set_value<double>;
        %template(get_value_d) get_value<double>;
        %template(update_buffer_d) update_buffer<double>;

        %template(get_value_i) get_value<long>;
        %template(set_value_i) set_value<long>;
        %template(update_buffer_i) update_buffer<long>;

        %template(get_value_c) get_value<unsigned char>;
        %template(set_value_c) set_value<unsigned char>;
        %template(update_buffer_c) update_buffer<unsigned char>;

    %pythoncode "port_extension.py"
}
