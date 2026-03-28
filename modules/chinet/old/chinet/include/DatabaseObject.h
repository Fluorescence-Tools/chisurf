#ifndef CHINET_DATABASEOBJECT_H
#define CHINET_DATABASEOBJECT_H

// This header provides a unified interface for database objects,
// whether using MongoDB or an in-memory implementation.

// Only include MongoObject.h when MongoDB is enabled
#ifdef WITH_MONGODB
// Use MongoDB implementation
#include "MongoObject.h"
typedef MongoObject DatabaseObject;
#else
// Use in-memory implementation
#include "MemoryObject.h"
typedef MemoryObject DatabaseObject;
#endif

#endif // CHINET_DATABASEOBJECT_H
