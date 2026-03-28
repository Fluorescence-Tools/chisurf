#include "info.h"
#include <cstdlib>
#include <string>
#include <cctype>

bool is_chinet_verbose() {
	const char* level_env = std::getenv("CHINET_LOG_LEVEL");
	if (level_env != nullptr) {
		std::string level(level_env);
		for (char &c : level) {
			c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
		}
		if (level == "DEBUG" || level == "INFO") {
			return true;
		}
		return false;
	}
	const char* env = std::getenv("CHINET_VERBOSE");
	return env != nullptr && std::string(env) != "";
}