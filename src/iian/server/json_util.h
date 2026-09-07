#pragma once
// Typed, validated JSON access with OpenAI-style error messages ("'temperature' must be a number").
#include "api_error.h"

#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace iian::server {

using json = nlohmann::json;

inline const char * json_type_name(const json & v) {
    switch (v.type()) {
        case json::value_t::null: return "null";
        case json::value_t::boolean: return "boolean";
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: return "integer";
        case json::value_t::number_float: return "number";
        case json::value_t::string: return "string";
        case json::value_t::array: return "array";
        case json::value_t::object: return "object";
        default: return "unknown";
    }
}

[[noreturn]] inline void json_type_error(const std::string & key, const char * expected, const json & got) {
    throw ApiError::bad_request("'" + key + "' must be " + expected + ", got " + json_type_name(got), key);
}

// Returns the field if present and non-null; missing/null -> nullopt; wrong type -> 400.
inline std::optional<double> json_number(const json & j, const std::string & key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    if (!it->is_number()) json_type_error(key, "a number", *it);
    return it->get<double>();
}
inline std::optional<int64_t> json_integer(const json & j, const std::string & key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    if (it->is_number_integer()) return it->get<int64_t>();
    if (it->is_number_float()) {
        double d = it->get<double>();
        if (d == (double) (int64_t) d) return (int64_t) d;
    }
    json_type_error(key, "an integer", *it);
}
inline std::optional<bool> json_bool(const json & j, const std::string & key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    if (!it->is_boolean()) json_type_error(key, "a boolean", *it);
    return it->get<bool>();
}
inline std::optional<std::string> json_string(const json & j, const std::string & key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    if (!it->is_string()) json_type_error(key, "a string", *it);
    return it->get<std::string>();
}
inline const json * json_array(const json & j, const std::string & key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return nullptr;
    if (!it->is_array()) json_type_error(key, "an array", *it);
    return &*it;
}
inline const json * json_object(const json & j, const std::string & key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return nullptr;
    if (!it->is_object()) json_type_error(key, "an object", *it);
    return &*it;
}

// "stop": "str" | ["a", "b"] -> vector
inline std::vector<std::string> json_string_or_list(const json & j, const std::string & key) {
    std::vector<std::string> out;
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return out;
    if (it->is_string()) { out.push_back(it->get<std::string>()); return out; }
    if (!it->is_array()) json_type_error(key, "a string or an array of strings", *it);
    for (const auto & e : *it) {
        if (!e.is_string()) json_type_error(key, "a string or an array of strings", e);
        out.push_back(e.get<std::string>());
    }
    return out;
}

inline std::vector<int32_t> json_int_list(const json & j, const std::string & key) {
    std::vector<int32_t> out;
    const json * arr = json_array(j, key);
    if (!arr) return out;
    for (const auto & e : *arr) {
        if (!e.is_number_integer()) json_type_error(key, "an array of integers", e);
        out.push_back(e.get<int32_t>());
    }
    return out;
}

inline json parse_json_body(const std::string & body) {
    if (body.empty()) throw ApiError::bad_request("request body is empty; send a JSON object");
    json j = json::parse(body, nullptr, /*allow_exceptions*/ false);
    if (j.is_discarded()) throw ApiError::bad_request("request body is not valid JSON");
    if (!j.is_object()) throw ApiError::bad_request("request body must be a JSON object");
    return j;
}

} // namespace iian::server
