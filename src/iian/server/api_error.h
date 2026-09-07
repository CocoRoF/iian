#pragma once
// OpenAI-shaped API errors: thrown from handlers, rendered as {"error": {...}} with an HTTP status.
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace iian::server {

struct ApiError : public std::runtime_error {
    int         status;
    std::string type;    // invalid_request_error | not_found_error | internal_error | authentication_error | permission_error | service_unavailable
    std::string param;   // offending parameter ("" -> null)
    std::string code;    // machine-readable code ("" -> null)

    ApiError(int status_, const std::string & message, std::string type_ = "invalid_request_error",
             std::string param_ = "", std::string code_ = "")
        : std::runtime_error(message), status(status_), type(std::move(type_)), param(std::move(param_)), code(std::move(code_)) {}

    nlohmann::json to_json() const {
        nlohmann::json e = {
            {"message", what()},
            {"type", type},
            {"param", param.empty() ? nlohmann::json(nullptr) : nlohmann::json(param)},
            {"code", code.empty() ? nlohmann::json(nullptr) : nlohmann::json(code)},
        };
        return nlohmann::json{{"error", e}};
    }

    static ApiError bad_request(const std::string & msg, const std::string & param = "") { return ApiError(400, msg, "invalid_request_error", param); }
    static ApiError unauthorized(const std::string & msg) { return ApiError(401, msg, "authentication_error", "", "invalid_api_key"); }
    static ApiError forbidden(const std::string & msg) { return ApiError(403, msg, "permission_error"); }
    static ApiError not_found(const std::string & msg, const std::string & param = "") { return ApiError(404, msg, "not_found_error", param, "model_not_found"); }
    static ApiError not_supported(const std::string & msg, const std::string & param = "") { return ApiError(400, msg, "invalid_request_error", param, "not_supported"); }
    static ApiError internal(const std::string & msg) { return ApiError(500, msg, "internal_error"); }
    static ApiError unavailable(const std::string & msg) { return ApiError(503, msg, "service_unavailable"); }
};

} // namespace iian::server
