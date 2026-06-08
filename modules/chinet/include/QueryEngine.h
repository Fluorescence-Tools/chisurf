#ifndef CHINET_QUERYENGINE_H
#define CHINET_QUERYENGINE_H

#include <functional>
#include "json.hpp"

using nlohmann::json;

class QueryEngine {
public:
    // Query operators matching MongoDB syntax
    static bool match(const json& document, const json& query) {
        for (const auto& [key, value] : query.items()) {
            if (key[0] == '$') {
                // Handle operators
                if (!handleOperator(key, document, value)) {
                    return false;
                }
            } else if (value.is_object() && !value.empty() && 
                      value.begin().key()[0] == '$') {
                // Field-level operators like {age: {$gte: 18}}
                if (!handleFieldOperators(key, document, value)) {
                    return false;
                }
            } else {
                // Direct equality check
                if (!document.contains(key) || document[key] != value) {
                    return false;
                }
            }
        }
        return true;
    }
    
private:
    static bool handleOperator(const std::string& op, const json& doc, const json& value) {
        if (op == "$and") {
            for (const auto& condition : value) {
                if (!match(doc, condition)) return false;
            }
            return true;
        } else if (op == "$or") {
            for (const auto& condition : value) {
                if (match(doc, condition)) return true;
            }
            return false;
        } else if (op == "$not") {
            return !match(doc, value);
        }
        return false;
    }
    
    static bool handleFieldOperators(const std::string& field, const json& doc, const json& operators) {
        if (!doc.contains(field)) return false;
        
        const auto& field_value = doc[field];
        
        for (const auto& [op, value] : operators.items()) {
            if (op == "$eq" && field_value != value) return false;
            if (op == "$ne" && field_value == value) return false;
            if (op == "$gt" && field_value <= value) return false;
            if (op == "$gte" && field_value < value) return false;
            if (op == "$lt" && field_value >= value) return false;
            if (op == "$lte" && field_value > value) return false;
            if (op == "$in") {
                bool found = false;
                for (const auto& v : value) {
                    if (field_value == v) {
                        found = true;
                        break;
                    }
                }
                if (!found) return false;
            }
            if (op == "$nin") {
                for (const auto& v : value) {
                    if (field_value == v) return false;
                }
            }
            if (op == "$regex") {
                // Add regex support
                std::regex pattern(value.get<std::string>());
                if (!std::regex_match(field_value.get<std::string>(), pattern)) {
                    return false;
                }
            }
            if (op == "$exists") {
                bool exists = doc.contains(field);
                if (exists != value.get<bool>()) return false;
            }
        }
        return true;
    }
};
