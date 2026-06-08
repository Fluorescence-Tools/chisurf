#ifndef CHINET_SCHEMAVALIDATOR_H
#define CHINET_SCHEMAVALIDATOR_H

#include <regex>
#include <functional>
#include "json.hpp"

using nlohmann::json;

class SchemaValidator {
public:
    struct ValidationResult {
        bool is_valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    struct FieldSchema {
        std::string type;  // "string", "number", "boolean", "object", "array"
        bool required = false;
        bool unique = false;
        json default_value;
        
        // Type-specific validations
        std::optional<size_t> min_length;
        std::optional<size_t> max_length;
        std::optional<double> min_value;
        std::optional<double> max_value;
        std::optional<std::string> pattern;  // Regex for strings
        std::optional<std::vector<json>> enum_values;
        
        // Custom validator
        std::function<bool(const json&)> custom_validator;
        
        // Nested schema for objects/arrays
        std::shared_ptr<json> nested_schema;
    };
    
private:
    std::unordered_map<std::string, json> schemas_;
    std::unordered_map<std::string, std::set<json>> unique_values_;
    
public:
    // Register a schema for a collection/type
    void registerSchema(const std::string& collection, const json& schema) {
        schemas_[collection] = schema;
    }
    
    // Validate document against schema
    ValidationResult validate(const std::string& collection, const json& document) {
        ValidationResult result{true, {}, {}};
        
        if (schemas_.find(collection) == schemas_.end()) {
            result.warnings.push_back("No schema defined for collection: " + collection);
            return result;
        }
        
        const json& schema = schemas_[collection];
        validateObject(document, schema, "", result);
        
        return result;
    }
    
private:
    void validateObject(const json& obj, const json& schema, 
                       const std::string& path, ValidationResult& result) {
        // Check required fields
        if (schema.contains("required")) {
            for (const auto& field : schema["required"]) {
                std::string field_name = field.get<std::string>();
                if (!obj.contains(field_name)) {
                    result.is_valid = false;
                    result.errors.push_back(path + field_name + " is required");
                }
            }
        }
        
        // Validate each field
        if (schema.contains("properties")) {
            for (const auto& [field, field_schema] : schema["properties"].items()) {
                if (obj.contains(field)) {
                    validateField(obj[field], field_schema, path + field + ".", result);
                }
            }
        }
        
        // Check for unknown fields
        if (schema.contains("additionalProperties") && 
            schema["additionalProperties"] == false) {
            for (const auto& [field, _] : obj.items()) {
                if (!schema["properties"].contains(field)) {
                    result.warnings.push_back(path + field + " is not defined in schema");
                }
            }
        }
    }
    
    void validateField(const json& value, const json& field_schema,
                       const std::string& path, ValidationResult& result) {
        // Type validation
        if (field_schema.contains("type")) {
            std::string expected_type = field_schema["type"];
            if (!checkType(value, expected_type)) {
                result.is_valid = false;
                result.errors.push_back(path + " must be of type " + expected_type);
                return;
            }
        }
        
        // String validations
        if (value.is_string()) {
            std::string str = value.get<std::string>();
            
            if (field_schema.contains("minLength")) {
                size_t min_len = field_schema["minLength"].get<size_t>();
                if (str.length() < min_len) {
                    result.is_valid = false;
                    result.errors.push_back(path + " must be at least " + 
                                           std::to_string(min_len) + " characters");
                }
            }
            
            if (field_schema.contains("maxLength")) {
                size_t max_len = field_schema["maxLength"].get<size_t>();
                if (str.length() > max_len) {
                    result.is_valid = false;
                    result.errors.push_back(path + " must be at most " + 
                                           std::to_string(max_len) + " characters");
                }
            }
            
            if (field_schema.contains("pattern")) {
                std::regex pattern(field_schema["pattern"].get<std::string>());
                if (!std::regex_match(str, pattern)) {
                    result.is_valid = false;
                    result.errors.push_back(path + " does not match pattern");
                }
            }
        }
        
        // Number validations
        if (value.is_number()) {
            double num = value.get<double>();
            
            if (field_schema.contains("minimum")) {
                double min_val = field_schema["minimum"].get<double>();
                if (num < min_val) {
                    result.is_valid = false;
                    result.errors.push_back(path + " must be >= " + std::to_string(min_val));
                }
            }
            
            if (field_schema.contains("maximum")) {
                double max_val = field_schema["maximum"].get<double>();
                if (num > max_val) {
                    result.is_valid = false;
                    result.errors.push_back(path + " must be <= " + std::to_string(max_val));
                }
            }
        }
        
        // Enum validation
        if (field_schema.contains("enum")) {
            bool found = false;
            for (const auto& enum_val : field_schema["enum"]) {
                if (value == enum_val) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                result.is_valid = false;
                result.errors.push_back(path + " must be one of the allowed values");
            }
        }
        
        // Array validations
        if (value.is_array() && field_schema.contains("items")) {
            for (size_t i = 0; i < value.size(); i++) {
                validateField(value[i], field_schema["items"], 
                            path + "[" + std::to_string(i) + "]", result);
            }
        }
        
        // Nested object validation
        if (value.is_object() && field_schema.contains("properties")) {
            validateObject(value, field_schema, path, result);
        }
    }
    
    bool checkType(const json& value, const std::string& type) {
        if (type == "string") return value.is_string();
        if (type == "number") return value.is_number();
        if (type == "integer") return value.is_number_integer();
        if (type == "boolean") return value.is_boolean();
        if (type == "object") return value.is_object();
        if (type == "array") return value.is_array();
        if (type == "null") return value.is_null();
        return false;
    }
};

// Schema-aware storage wrapper
class ValidatedStorage {
private:
    std::shared_ptr<InMemoryStorage> storage_;
    std::shared_ptr<SchemaValidator> validator_;
    bool strict_mode_ = false;  // Reject invalid documents
    
public:
    ValidatedStorage(std::shared_ptr<InMemoryStorage> storage, bool strict = false)
        : storage_(storage), 
          validator_(std::make_shared<SchemaValidator>()),
          strict_mode_(strict) {}
    
    bool store(const std::string& collection, const std::string& key, const json& value) {
        auto validation = validator_->validate(collection, value);
        
        if (!validation.is_valid && strict_mode_) {
            // Log errors
            for (const auto& error : validation.errors) {
                std::cerr << "Validation error: " << error << std::endl;
            }
            return false;
        }
        
        // Apply defaults and transformations
        json processed_value = applyDefaults(collection, value);
        
        return storage_->store(collection + ":" + key, processed_value);
    }
    
    void defineSchema(const std::string& collection, const json& schema) {
        validator_->registerSchema(collection, schema);
    }
    
private:
    json applyDefaults(const std::string& collection, const json& value) {
        // Apply default values for missing fields
        json result = value;
        
        // Implementation would check schema and apply defaults
        
        return result;
    }
};

#endif // CHINET_SCHEMAVALIDATOR_H
