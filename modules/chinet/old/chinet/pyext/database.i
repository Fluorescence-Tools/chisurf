%module(directors="1") chinet

%{
#include "../include/DatabaseObject.h"
#ifdef WITH_MONGODB
  #include "../include/MongoObject.h"
#else
  #include "../include/MemoryObject.h"
#endif
%}

/* Tell SWIG about the backend types, but don’t wrap them twice */
#ifdef WITH_MONGODB
  %import "../include/MongoObject.h"
  %shared_ptr(MongoObject)
#else
  %import "../include/MemoryObject.h"
  %shared_ptr(MemoryObject)
#endif

/* Create DatabaseObject as a C++ typedef of the active backend */
%inline %{
#ifdef WITH_MONGODB
  typedef MongoObject DatabaseObject;
#else
  typedef MemoryObject DatabaseObject;
#endif
%}

/* Now wrap only the top‐level header */
%include "../include/DatabaseObject.h"

/* Expose shared_ptr and common templates under the new name */
%shared_ptr(DatabaseObject)
%template(ListDatabaseObjectPtr) std::list<std::shared_ptr<DatabaseObject>>;

#ifdef WITH_MONGODB
  /* MongoObject-specific attributes and extensions */
  %attributestring(MongoObject, std::string, name,    get_name,    set_name);
  %attributestring(MongoObject, std::string, oid,     get_own_oid, set_own_oid);
  %attribute(MongoObject, bool,          is_connected_to_db, is_connected_to_db);

  %extend MongoObject {
    %template(get_array_double)     get_array<double>;
    %template(set_array_double)     set_array<double>;
    %template(get_array_long)       get_array<long>;
    %template(set_array_long)       set_array<long>;
    %template(get_array_int)        get_array<int>;
    %template(set_array_int)        set_array<int>;

    %template(get_singleton_double) get_singleton<double>;
    %template(set_singleton_double) set_singleton<double>;
    %template(get_singleton_int)    get_singleton<int>;
    %template(set_singleton_int)    set_singleton<int>;
    %template(get_singleton_bool)   get_singleton<bool>;
    %template(set_singleton_bool)   set_singleton<bool>;

    %template(connect_object_to_db)
      connect_object_to_db<std::shared_ptr<MongoObject>>;

    std::shared_ptr<MongoObject> __getitem__(char* key) {
      return (*self)[key];
    }

    %pythoncode "database_extension.py"
  }
#else
  /* MemoryObject-specific attributes and extensions */
  %attributestring(MemoryObject, std::string, name,    get_name,    set_name);
  %attributestring(MemoryObject, std::string, oid,     get_own_oid, set_own_oid);
  %attribute(MemoryObject, bool,          is_connected_to_db, is_connected_to_db);

  %extend MemoryObject {
    %template(get_array_double)     get_array<double>;
    %template(set_array_double)     set_array<double>;
    %template(get_array_long)       get_array<long>;
    %template(set_array_long)       set_array<long>;
    %template(get_array_int)        get_array<int>;
    %template(set_array_int)        set_array<int>;

    %template(get_singleton_double) get_singleton<double>;
    %template(set_singleton_double) set_singleton<double>;
    %template(get_singleton_int)    get_singleton<int>;
    %template(set_singleton_int)    set_singleton<int>;
    %template(get_singleton_bool)   get_singleton<bool>;
    %template(set_singleton_bool)   set_singleton<bool>;

    %template(connect_object_to_db)
      connect_object_to_db<std::shared_ptr<MemoryObject>>;

    std::shared_ptr<MemoryObject> __getitem__(char* key) {
      return (*self)[key];
    }

    %pythoncode "database_extension.py"
  }
#endif
