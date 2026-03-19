#include "info.h"
#include <cstdlib>
#include <string>

bool is_chinet_verbose() {
    const char* env = std::getenv("CHINET_VERBOSE");
    return env != nullptr && std::string(env) != "";
}