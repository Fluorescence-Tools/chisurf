%module(directors="1", package="chinet") chinet
%feature("kwargs", 1);

#ifdef CHINET_VERBOSE
// Warning 509: Overloaded method ignored,
// Warning 511: Ignore overloaded functions
// Warning 401: Nothing known about base class
// Warning 453: No typemaps are defined. (Some typemaps are defined and not used)
// Warning 319: No access specifier given for base class (std::enable_shared_from_this)
// Warning 362: operator= ignored
#pragma SWIG nowarn=314,319,503,511,401,389
#endif

%include "documentation.i"
%{
    #define SWIG_FILE_WITH_INIT

    #include "../include/Functions.h"
    #include "../include/CNode.h"
    #include "../include/MongoObject.h"
%}

%include "numpy.i"
%init %{
    import_array();
%}

%pythonbegin "python_extension.py"

%include "typemaps.i";
%include "stl.i";
%include "std_wstring.i";
%include "std_string.i";
%include "std_map.i";
%include "std_vector.i";
%include "std_list.i";
%include "std_shared_ptr.i";
%include "cpointer.i"
%include "std_set.i";
%include attribute.i

%template(MapStringString) std::map<std::string, std::string>;
%template(VectorString) std::vector<std::string>;
%template(VectorDouble) std::vector<double>;
%template(VectorInt) std::vector<int>;
%template(VectorLong) std::vector<long>;
%template(MapStringVectorDouble) std::map<std::string, std::vector<double>>;


%extend std::vector<double>{

    string __str__(){
        std::ostringstream os;
        os << "Double Vector: [ ";
        int len = $self->size();
        if (len > 100){ // only print the first and last few
            for (int i=0 ; i < 5; i++){
                os << $self->at(i) << ", ";
            }
            os << " ..... ";
            for (int i=len-5 ; i < len; i++){
                os << $self->at(i) << ", ";
            }
        }
        else{
            for (int i=0 ; i < len; i++){
                os << $self->at(i) << ", ";
            }
        }
        os << "]";
        return os.str();
    }

    string __repr__(){
        std::ostringstream os;
        os << "DoubleVector( [";
        for (int i=0 ; i < $self->size(); i++){
            os << $self->at(i) << ", ";
        }
        os << "] )";
        return os.str();
    }

}

%include "../include/info.h"
%include "mongo.i"
%include "port.i"
%include "node.i"
%include "session.i"
