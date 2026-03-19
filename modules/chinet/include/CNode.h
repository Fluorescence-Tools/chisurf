/// @file CNode.h
/// @brief This file declares the Node class, which represents a computational node
///        with input/output ports, callback functionality, and database interaction.

#ifndef chinet_Node_H
#define chinet_Node_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include "DatabaseObject.h"
#include "Port.h"
#include "NodeCallback.h"

// Forward define classes
class Port;
class NodeCallback;

/// @class Node
/// @brief Represents a computational node with ports and callbacks.
///
/// The `Node` class is derived from `DatabaseObject` and allows interaction
/// with input and output ports, evaluation of node functionality,
/// and database communication. A callback mechanism is provided via
/// `NodeCallback`.
class Node : public DatabaseObject {

private:
    friend Port; ///< Declares `Port` as a friend class to enable access to private members.

    bool node_valid_ = false; ///< Indicates whether the node is valid.
    std::map<std::string, std::shared_ptr<Port>> ports; ///< Holds all ports (input/output) in a map.

protected:
    /// @brief Populates input and output port lookups.
    void fill_input_output_port_lookups();

    std::map<std::string, std::shared_ptr<Port>> in_;  ///< Lookup table for input ports.
    std::map<std::string, std::shared_ptr<Port>> out_; ///< Lookup table for output ports.
    std::string callback; ///< Holds the callback name.
    std::string callback_type_string; ///< Indicates the type of the callback as a string.
    int callback_type = -1; ///< Callback type as an integer. -1 means no callback configured.

public:
    /// A shared pointer managing a callback class for this node.
    std::shared_ptr<NodeCallback> callback_class;

    /// @name Constructor & Destructor
    /// @{
    /// @brief Constructs a new `Node` object.
    /// @param name The name of the node (optional, default = "").
    /// @param ports Map of port names to `Port` objects (optional).
    /// @param callback_class Shared pointer to a `NodeCallback` object (optional).
    Node(
            std::string name = "",
            const std::map<std::string, std::shared_ptr<Port>>& ports = std::map<std::string, std::shared_ptr<Port>>(),
            std::shared_ptr<NodeCallback> callback_class = nullptr
    );

    /// @brief Destroys the `Node` object.
    ~Node();
    /// @}

    /// @name Methods
    /// @{
    /// @brief Reads the node's data from the database by its object ID.
    /// @param oid_string The object ID (OID) of the node in the database.
    /// @return `true` if the operation succeeds, `false` otherwise.
    bool read_from_db(const std::string &oid_string) final;

    /// @brief Performs evaluation of the node.
    void evaluate();

    /// @brief Checks whether the node is valid.
    /// @return `true` if the node is valid, otherwise `false`.
    bool is_valid();

    /// @brief Checks the validity of the node's input ports.
    /// @return `true` if all input ports are valid, otherwise `false`.
    bool inputs_valid();

    /// @brief Writes the node's data to the database.
    /// @return `true` if the operation succeeds, `false` otherwise.
    bool write_to_db() final;
    /// @}

    /// @name Getter Methods
    /// @{
    /// @brief Retrieves a BSON representation of the node.
    /// @return A BSON object representing the node's properties.
#ifdef WITH_MONGODB
    bson_t get_bson() final;
#endif

    /// @brief Retrieves the name of the node.
    /// @return A string containing the node's name.
    std::string get_name();

    /// @brief Gets all input ports of the node.
    /// @return A map containing input ports with their names as keys.
    const std::map<std::string, std::shared_ptr<Port>>& get_input_ports() const;

    /// @brief Gets all output ports of the node.
    /// @return A map containing output ports with their names as keys.
    const std::map<std::string, std::shared_ptr<Port>>& get_output_ports() const;

    /// @brief Gets all ports of the node (both input and output).
    /// @return A map containing all ports with their names as keys.
    const std::map<std::string, std::shared_ptr<Port>>& get_ports() const;

    /// @brief Retrieves a port by its name.
    /// @param port_name The name of the port to retrieve.
    /// @return A shared pointer to the requested `Port` object.
    std::shared_ptr<Port> get_port(const std::string &port_name);

    /// @brief Retrieves an input port by its name.
    /// @param port_name The name of the input port.
    /// @return A shared pointer to the requested input `Port` object.
    std::shared_ptr<Port> get_input_port(const std::string &port_name);

    /// @brief Retrieves an output port by its name.
    /// @param port_name The name of the output port.
    /// @return A shared pointer to the requested output `Port` object.
    std::shared_ptr<Port> get_output_port(const std::string &port_name);
    /// @}

    /// @name Setter Methods
    /// @{
    /// @brief Configures all ports of a node.
    /// @param ports Map of port names to their respective `Port` objects.
    void set_ports(const std::map<std::string, std::shared_ptr<Port>>& ports);

    /// @brief Adds a port to the node.
    /// @param key The name of the port.
    /// @param port A shared pointer to the `Port` object.
    /// @param is_source Indicates whether the port is a source port.
    /// @param fill_in_out Determines whether input/output lookups should be updated.
    void add_port(
            const std::string &key,
            std::shared_ptr<Port> port,
            bool is_source,
            bool fill_in_out = true
    );

    /// @brief Adds an input port to the node.
    /// @param key The name of the input port.
    /// @param port A shared pointer to the input `Port` object.
    void add_input_port(const std::string &key, std::shared_ptr<Port> port);

    /// @brief Adds an output port to the node.
    /// @param key The name of the output port.
    /// @param port A shared pointer to the output `Port` object.
    void add_output_port(const std::string &key, std::shared_ptr<Port> port);

    /// @brief Sets the callback for the node.
    /// @param callback The callback's name as a string.
    /// @param callback_type The type of the callback as a string.
    void set_callback(std::string callback, std::string callback_type);

    /// @brief Sets the callback of the node using a `NodeCallback` object.
    /// @param cb Shared pointer to the `NodeCallback` object.
    void set_callback(std::shared_ptr<NodeCallback> cb);

    /// @brief Sets the validity of the node.
    /// @param is_valid A boolean indicating if the node is valid.
    void set_valid(bool is_valid = false);
    /// @}
};

#endif // chinet_Node_H
