#include "iian/arch.h"
namespace iian {
// Architectures self-register via static initializers in arch/*.cpp. Because the library is static,
// we reference a symbol per arch file to keep the linker from dropping them.
void register_builtin_archs() {
    // The static registrars run at load time; this function exists to force-link the object files.
}
} // namespace iian
